import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export

# Shared by `Mistral3PatchMerger`, `Mistral3MultiModalProjector`, and
# `MistralCausalLMPreprocessor` (which must expand exactly this many `[IMG]`
# placeholder tokens per merged patch). Defined once here so the model and
# preprocessor defaults cannot silently drift apart.
MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE = 2


def _mistral_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


class Mistral3ImageFeatureExtractor(keras.layers.Layer):
    """Computes projected Mistral3 image features.

    `MistralBackbone` needs to run this computation on `keras.Input`
    placeholders while building its functional graph. Doing that as a
    plain function called directly on `KerasTensor`s (rather than inside a
    `Layer.call()`) breaks: shape queries like `ops.shape(image_sizes)[0]`
    return a placeholder `None` for the dynamic batch dimension at
    graph-construction time (rather than a real or traced integer, as they
    would inside an actual `call()`), and `None` then propagates into
    tensor arithmetic that expects a number. Wrapping the logic in a layer
    — with `compute_output_spec` declaring the output shape — means it
    only ever runs against real backend tensors, during an actual forward
    pass.

    Args:
        vision_encoder: A `Mistral3VisionEncoder` instance.
        multimodal_projector: A `Mistral3MultiModalProjector` instance.
        vision_feature_layer: int. Which vision encoder hidden state to
            project. Only the final hidden state (`-1`) is currently
            supported, matching every published Mistral3 configuration.
            Any other value raises `NotImplementedError`, since
            `Mistral3VisionEncoder` only returns its final hidden state
            (it does not collect intermediate per-layer hidden states).
    """

    def __init__(
        self,
        vision_encoder,
        multimodal_projector,
        vision_feature_layer=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vision_feature_layer not in (-1, "last"):
            raise NotImplementedError(
                "`vision_feature_layer` only supports `-1` (the final "
                "vision hidden state). `Mistral3VisionEncoder` does not "
                f"expose intermediate hidden states. Received: "
                f"{vision_feature_layer}."
            )
        self.vision_encoder = vision_encoder
        self.multimodal_projector = multimodal_projector
        self.vision_feature_layer = vision_feature_layer

    def call(self, pixel_values, image_sizes, training=None):
        image_features = self.vision_encoder(
            pixel_values,
            image_sizes=image_sizes,
            training=training,
        )
        # `vision_encoder` concatenates all images into a single batch-1
        # sequence of exactly `sum(patch_counts)` tokens, in the same
        # per-image order/offsets as `image_sizes`. That is already the
        # "real content" prefix `Mistral3MultiModalProjector` expects — it
        # just needs zero-padding at the tail up to the padded-canvas size
        # `num_images * max_patch_height * max_patch_width` so the
        # projector can operate on a fixed-shape buffer.
        image_features = ops.squeeze(image_features, axis=0)

        image_sizes_int = ops.cast(ops.convert_to_tensor(image_sizes), "int32")
        patch_size = self.multimodal_projector.patch_size
        patch_heights = image_sizes_int[:, 0] // patch_size
        patch_widths = image_sizes_int[:, 1] // patch_size
        max_patch_height = ops.max(patch_heights)
        max_patch_width = ops.max(patch_widths)
        num_images = ops.shape(image_sizes)[0]
        padded_total = num_images * max_patch_height * max_patch_width
        pad_amount = padded_total - ops.shape(image_features)[0]
        image_features = ops.pad(image_features, [[0, pad_amount], [0, 0]])

        return self.multimodal_projector(
            image_features,
            image_sizes=image_sizes,
            max_patch_height=max_patch_height,
            max_patch_width=max_patch_width,
        )

    def compute_output_spec(self, pixel_values, image_sizes, training=None):
        return keras.KerasTensor(
            shape=(None, self.multimodal_projector.text_hidden_dim),
            dtype=self.compute_dtype,
        )


def compute_image_placeholder_indices(token_ids, image_token_index):
    """Compute flat image placeholder indices for `token_ids` on the host.

    `Mistral3ImageTextEmbeddingMerger` needs the flat positions of image
    placeholder tokens to scatter projected image features into. Finding
    those positions requires a data-dependent (`nonzero`-style) op whose
    output shape depends on the number of placeholder tokens actually
    present — a shape that isn't known until the values are inspected.
    That is incompatible with `jax.jit` tracing (used by Keras's JAX
    backend to compile `predict_step`/`train_step`/`generate`), which
    requires all shapes to be static at trace time.

    This helper does that computation eagerly with NumPy, outside of any
    jitted graph — e.g. in a preprocessing step, before `token_ids` is fed
    to the model. The resulting indices are then passed to the model as a
    plain tensor input, so the layer itself only performs a static-shape
    `scatter_update`.

    Args:
        token_ids: int array `(batch, seq_length)`.
        image_token_index: int. The token ID marking image placeholder
            positions.

    Returns:
        int32 NumPy array `(num_placeholders,)` of flat indices into the
        flattened `(batch * seq_length,)` sequence.
    """
    token_ids = np.asarray(token_ids)
    flat_token_ids = np.reshape(token_ids, (-1,))
    return np.nonzero(flat_token_ids == image_token_index)[0].astype("int32")


def compute_pixtral_resize_size(height, width, longest_edge, patch_size):
    """Computes the HF `PixtralImageProcessor` resize target for one image.

    Scales `(height, width)` down (preserving aspect ratio) so its longest
    edge is at most `longest_edge`, then rounds each dimension up to the
    nearest multiple of `patch_size`.

    Args:
        height: int. The image's original height.
        width: int. The image's original width.
        longest_edge: int. The maximum allowed size of the longer side.
        patch_size: int. The patch size each output dimension must be a
            multiple of.

    Returns:
        `(resized_height, resized_width)` as plain Python ints.
    """
    ratio = max(height / longest_edge, width / longest_edge)
    if ratio > 1:
        height = int(height / ratio)
        width = int(width / ratio)

    resized_height = ((height - 1) // patch_size + 1) * patch_size
    resized_width = ((width - 1) // patch_size + 1) * patch_size
    return resized_height, resized_width


class Mistral3ImageTextEmbeddingMerger(keras.layers.Layer):
    """Scatters projected image features into image placeholder positions.

    Replaces the token embeddings at `placeholder_indices` with the
    concatenated, projected image features, matching HF's
    `masked_scatter` fusion in `Mistral3Model`.

    `placeholder_indices` (the flat positions of image placeholder tokens)
    must be precomputed outside this layer — see
    `compute_image_placeholder_indices` — since deriving them here via a
    `nonzero`-style op would make the layer incompatible with `jax.jit`
    tracing.
    """

    def call(self, token_embeddings, image_features, placeholder_indices):
        batch_size = ops.shape(token_embeddings)[0]
        seq_length = ops.shape(token_embeddings)[1]
        hidden_dim = ops.shape(token_embeddings)[2]

        flat_embeddings = ops.reshape(
            token_embeddings,
            (batch_size * seq_length, hidden_dim),
        )

        # `placeholder_indices` arrives as a flat `(total_placeholders,)`
        # tensor from an eager caller, or as a batched `(batch, N)` tensor
        # once passed through a `keras.Input` — flatten to match
        # `flat_embeddings`. Values are global indices into the flattened
        # `(batch * seq_length,)` sequence either way.
        if len(ops.shape(placeholder_indices)) == 2:
            placeholder_indices = ops.reshape(placeholder_indices, (-1,))
        placeholder_indices = ops.expand_dims(
            ops.cast(placeholder_indices, "int32"),
            axis=-1,
        )

        image_features = ops.cast(image_features, flat_embeddings.dtype)
        merged_embeddings = ops.scatter_update(
            inputs=flat_embeddings,
            indices=placeholder_indices,
            updates=image_features,
        )

        return ops.reshape(
            merged_embeddings,
            (batch_size, seq_length, hidden_dim),
        )

    def compute_output_shape(self, input_shape):
        return input_shape


class Mistral3VisionRotaryEmbedding(keras.layers.Layer):
    """
    Pixtral's 2D rotary positional embedding.

    Unlike the text Mistral RoPE, Pixtral constructs frequencies from
    2D patch coordinates. The first half corresponds to height and the
    second half to width.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        head_dim,
        rope_theta=10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.max_patches_per_side = image_size // patch_size

    def _create_inv_freq(self):
        max_patches = self.max_patches_per_side

        freq_indices = np.arange(0, self.head_dim, 2, dtype="float32")
        freqs = np.power(
            self.rope_theta,
            -freq_indices / self.head_dim,
        )

        height_indices = np.arange(max_patches, dtype="float32")
        width_indices = np.arange(max_patches, dtype="float32")

        freqs_h = np.einsum(
            "i,j->ij",
            height_indices,
            freqs[::2],
        )

        freqs_w = np.einsum(
            "i,j->ij",
            width_indices,
            freqs[1::2],
        )

        # [H, 1, D/4] -> [H, W, D/4]
        freqs_h = np.broadcast_to(
            np.expand_dims(freqs_h, axis=1),
            (max_patches, max_patches, freqs_h.shape[-1]),
        )

        # [1, W, D/4] -> [H, W, D/4]
        freqs_w = np.broadcast_to(
            np.expand_dims(freqs_w, axis=0),
            (max_patches, max_patches, freqs_w.shape[-1]),
        )

        inv_freq = np.concatenate(
            [freqs_h, freqs_w],
            axis=-1,
        )

        inv_freq = np.reshape(
            inv_freq,
            (-1, self.head_dim // 2),
        )

        return np.concatenate(
            [inv_freq, inv_freq],
            axis=-1,
        )

    def build(self, input_shape):
        max_patches = self.max_patches_per_side
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(max_patches**2, self.head_dim),
            initializer=keras.initializers.Constant(self._create_inv_freq()),
            trainable=False,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "head_dim": self.head_dim,
                "rope_theta": self.rope_theta,
            }
        )
        return config

    def call(self, position_ids, dtype=None):
        freqs = ops.take(
            self.inv_freq,
            position_ids,
            axis=0,
        )

        cos = ops.cos(freqs)
        sin = ops.sin(freqs)

        if dtype is not None:
            cos = ops.cast(cos, dtype)
            sin = ops.cast(sin, dtype)

        return cos, sin


def _rotate_half(x):
    half = ops.shape(x)[-1] // 2

    x1 = x[..., :half]
    x2 = x[..., half:]

    return ops.concatenate(
        [-x2, x1],
        axis=-1,
    )


def _apply_rotary_pos_emb(q, k, cos, sin):
    # q/k: [B, H, S, D]; cos/sin: [B, S, D].
    # Expand at axis 1 to broadcast cos/sin across attention heads.

    cos = ops.expand_dims(cos, axis=1)
    sin = ops.expand_dims(sin, axis=1)

    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    return q, k


class Mistral3VisionAttention(keras.layers.Layer):
    """Multi-head self-attention used by Pixtral.

    Args:
        hidden_dim: int. The size of the attention layer's input/output.
        num_heads: int. The number of attention heads.
        head_dim: int. The size of each attention head. Defaults to
            `hidden_dim // num_heads`.
        dropout: float. The dropout probability applied to attention scores.
            Defaults to `0.0`.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        head_dim=None,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_dim // num_heads
        self.dropout = dropout

        self.scaling = self.head_dim**-0.5

        self.q_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="v_proj",
        )
        self.o_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="o_proj",
        )

        self.attention_dropout = keras.layers.Dropout(dropout)

    def _reshape_to_heads(self, x):
        batch_size = ops.shape(x)[0]
        sequence_length = ops.shape(x)[1]

        x = ops.reshape(
            x,
            (
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ),
        )

        return ops.transpose(x, (0, 2, 1, 3))

    def call(
        self,
        inputs,
        attention_mask=None,
        position_embeddings=None,
        training=None,
    ):
        q = self._reshape_to_heads(self.q_proj(inputs))
        k = self._reshape_to_heads(self.k_proj(inputs))
        v = self._reshape_to_heads(self.v_proj(inputs))

        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # [B, H, S, D] @ [B, H, D, S]
        attention_scores = ops.matmul(
            q,
            ops.transpose(k, (0, 1, 3, 2)),
        )

        attention_scores = attention_scores * self.scaling

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # HF explicitly performs softmax in float32.
        attention_scores = ops.cast(
            attention_scores,
            "float32",
        )

        attention_scores = ops.softmax(
            attention_scores,
            axis=-1,
        )

        attention_scores = ops.cast(
            attention_scores,
            q.dtype,
        )

        attention_scores = self.attention_dropout(
            attention_scores,
            training=training,
        )

        attention_output = ops.matmul(
            attention_scores,
            v,
        )

        attention_output = ops.transpose(
            attention_output,
            (0, 2, 1, 3),
        )

        attention_output = ops.reshape(
            attention_output,
            (
                ops.shape(attention_output)[0],
                ops.shape(attention_output)[1],
                self.hidden_dim,
            ),
        )

        return self.o_proj(attention_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "dropout": self.dropout,
            }
        )
        return config


class Mistral3VisionMLP(keras.layers.Layer):
    """SwiGLU MLP used by Pixtral.

    Args:
        hidden_dim: int. The size of the MLP's input/output.
        intermediate_dim: int. The size of the MLP's intermediate layer.
        activation: str or callable. The activation applied to the gate
            projection. Defaults to `"silu"`.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation

        self.gate_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="gate_proj",
        )

        self.up_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="up_proj",
        )

        self.down_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="down_proj",
        )

        self.activation_fn = keras.activations.get(activation)

    def call(self, inputs):
        gate = self.activation_fn(self.gate_proj(inputs))
        up = self.up_proj(inputs)

        return self.down_proj(gate * up)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
            }
        )
        return config


class Mistral3VisionEncoderLayer(keras.layers.Layer):
    """One Pixtral transformer encoder layer.

    Args:
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The size of the MLP's intermediate layer.
        num_heads: int. The number of attention heads.
        head_dim: int. The size of each attention head. Defaults to
            `hidden_dim // num_heads`.
        layer_norm_epsilon: float. The epsilon for RMS normalization.
            Defaults to `1e-5`.
        activation: str or callable. The MLP activation. Defaults to
            `"silu"`.
        dropout: float. The attention dropout probability. Defaults to
            `0.0`.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        head_dim=None,
        layer_norm_epsilon=1e-5,
        activation="silu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_dim // num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation = activation
        self.dropout = dropout

        self.attention_norm = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            name="attention_norm",
        )

        self.attention = Mistral3VisionAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            name="attention",
        )

        self.ffn_norm = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            name="ffn_norm",
        )

        self.feed_forward = Mistral3VisionMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation=activation,
            name="feed_forward",
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        training=None,
    ):
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)

        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            training=training,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "activation": self.activation,
                "dropout": self.dropout,
            }
        )
        return config


@keras_hub_export("keras_hub.models.Mistral3VisionEncoder")
class Mistral3VisionEncoder(keras.Model):
    """
    Pixtral vision encoder used by Mistral3.

    This is not exposed as a standalone Pixtral model. It is the vision
    tower consumed by the Mistral3 multimodal architecture.

    `pixel_values` follows the Hugging Face Pixtral layout:
    `(num_images, num_channels, height, width)`.

    When `image_sizes` is provided, each image is cropped to its effective
    image size after patchification, the resulting patch sequences are
    concatenated into a single sequence, and attention is restricted to
    patches belonging to the same image.
    """

    def __init__(
        self,
        image_size=1540,
        patch_size=14,
        num_channels=3,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        head_dim=64,
        intermediate_dim=4096,
        rope_theta=10000.0,
        layer_norm_epsilon=1e-5,
        activation="silu",
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.rope_theta = rope_theta
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation = activation
        self.attention_dropout = attention_dropout

        self.patch_conv = keras.layers.Conv2D(
            filters=hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            kernel_initializer=_mistral_kernel_initializer(),
            name="patch_conv",
        )

        self.ln_pre = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            name="ln_pre",
        )

        self.patch_positional_embedding = Mistral3VisionRotaryEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            name="patch_positional_embedding",
        )

        self.transformer_layers = []

        for i in range(num_layers):
            self.transformer_layers.append(
                Mistral3VisionEncoderLayer(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    layer_norm_epsilon=layer_norm_epsilon,
                    activation=activation,
                    dropout=attention_dropout,
                    name=f"transformer_layer_{i}",
                )
            )

    def _create_position_ids_for_images(
        self,
        image_sizes,
        max_patch_height,
        max_patch_width,
    ):
        """Create mesh-grid position IDs using tensor image sizes."""
        patch_heights = image_sizes[:, 0] // self.patch_size
        patch_widths = image_sizes[:, 1] // self.patch_size
        num_images = ops.shape(image_sizes)[0]

        height_indices = ops.arange(max_patch_height, dtype="int32")
        width_indices = ops.arange(max_patch_width, dtype="int32")
        height_grid = ops.expand_dims(height_indices, axis=1)
        width_grid = ops.expand_dims(width_indices, axis=0)
        position_grid = (
            height_grid * (self.image_size // self.patch_size) + width_grid
        )
        position_grid = ops.broadcast_to(
            position_grid,
            (num_images, max_patch_height, max_patch_width),
        )

        height_valid = ops.expand_dims(
            ops.expand_dims(height_indices, axis=0)
            < ops.expand_dims(patch_heights, axis=1),
            axis=2,
        )
        width_valid = ops.expand_dims(
            ops.expand_dims(width_indices, axis=0)
            < ops.expand_dims(patch_widths, axis=1),
            axis=1,
        )
        valid_patches = ops.logical_and(height_valid, width_valid)
        position_grid = ops.reshape(position_grid, (-1,))
        valid_patches = ops.reshape(valid_patches, (-1,))
        valid_indices = ops.nonzero(valid_patches)[0]
        return ops.take(position_grid, valid_indices, axis=0)

    def _create_block_attention_mask(self, patch_counts, dtype):
        """Create a block-diagonal attention mask from tensor patch counts."""
        patch_counts = ops.cast(patch_counts, "int32")
        num_images = ops.shape(patch_counts)[0]
        block_ids = ops.repeat(
            ops.arange(num_images, dtype="int32"),
            patch_counts,
        )
        same_block = ops.equal(
            ops.expand_dims(block_ids, axis=0),
            ops.expand_dims(block_ids, axis=1),
        )

        dtype = keras.backend.standardize_dtype(dtype)
        if dtype == "float16":
            mask_value = -65504.0
        elif dtype == "bfloat16":
            mask_value = -3.38953139e38
        else:
            mask_value = -3.4028234663852886e38

        mask = ops.where(
            same_block,
            ops.zeros_like(ops.cast(same_block, dtype)),
            ops.cast(mask_value, dtype),
        )
        return ops.expand_dims(ops.expand_dims(mask, axis=0), axis=0)

    def _normalize_image_sizes(self, image_sizes, num_images, height, width):
        """Normalize image sizes as an integer tensor."""
        if image_sizes is None:
            image_sizes = ops.stack([height, width])
            image_sizes = ops.broadcast_to(
                image_sizes,
                (num_images, 2),
            )
        else:
            image_sizes = ops.convert_to_tensor(image_sizes)

        return ops.cast(image_sizes, "int32")

    def _extract_patch_sequences(self, patch_embeds, image_sizes):
        """Crop and flatten patch embeddings with a tensor mask."""
        image_sizes = ops.cast(image_sizes, "int32")
        patch_heights = image_sizes[:, 0] // self.patch_size
        patch_widths = image_sizes[:, 1] // self.patch_size
        max_patch_height = ops.shape(patch_embeds)[1]
        max_patch_width = ops.shape(patch_embeds)[2]

        height_indices = ops.arange(max_patch_height, dtype="int32")
        width_indices = ops.arange(max_patch_width, dtype="int32")
        height_valid = ops.expand_dims(
            ops.expand_dims(height_indices, axis=0)
            < ops.expand_dims(patch_heights, axis=1),
            axis=2,
        )
        width_valid = ops.expand_dims(
            ops.expand_dims(width_indices, axis=0)
            < ops.expand_dims(patch_widths, axis=1),
            axis=1,
        )
        valid_patches = ops.logical_and(height_valid, width_valid)
        patch_embeds = ops.reshape(
            patch_embeds,
            (-1, ops.shape(patch_embeds)[-1]),
        )
        valid_patches = ops.reshape(valid_patches, (-1,))
        valid_indices = ops.nonzero(valid_patches)[0]
        return ops.take(patch_embeds, valid_indices, axis=0)

    def call(
        self,
        pixel_values,
        image_sizes=None,
        training=None,
    ):
        # HF input layout: [num_images, channels, height, width].
        pixel_values = ops.transpose(pixel_values, (0, 2, 3, 1))
        pixel_values = ops.cast(
            pixel_values,
            self.patch_conv.variable_dtype,
        )
        patch_embeds = self.patch_conv(pixel_values)
        max_patch_height = ops.shape(patch_embeds)[1]
        max_patch_width = ops.shape(patch_embeds)[2]

        image_sizes = self._normalize_image_sizes(
            image_sizes,
            num_images=ops.shape(pixel_values)[0],
            height=ops.shape(pixel_values)[1],
            width=ops.shape(pixel_values)[2],
        )
        patch_embeds = self._extract_patch_sequences(
            patch_embeds,
            image_sizes,
        )
        patch_embeds = ops.expand_dims(patch_embeds, axis=0)
        patch_embeds = self.ln_pre(patch_embeds)

        position_ids = self._create_position_ids_for_images(
            image_sizes,
            max_patch_height,
            max_patch_width,
        )
        position_ids = ops.expand_dims(position_ids, axis=0)
        cos, sin = self.patch_positional_embedding(
            position_ids,
            dtype=patch_embeds.dtype,
        )

        patch_counts = (image_sizes[:, 0] // self.patch_size) * (
            image_sizes[:, 1] // self.patch_size
        )
        attention_mask = self._create_block_attention_mask(
            patch_counts,
            patch_embeds.dtype,
        )

        hidden_states = patch_embeds
        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                training=training,
            )
        return hidden_states

    def compute_output_spec(
        self, pixel_values, image_sizes=None, training=None
    ):
        """Return the output shape spec without running the forward pass.

        Needed because `MistralBackbone` wraps this encoder inside a Keras
        functional model with a dynamic-shape (`None` height/width)
        `pixel_values` input, so Keras must infer the output shape at graph
        construction time. Actually tracing `call()` requires resolving
        data-dependent shape ops (`ops.nonzero`, dynamic `ops.repeat`) on
        symbolic dimensions, which the JAX backend's functional-graph
        tracer cannot do (`InconclusiveDimensionOperation`). Declaring the
        shape explicitly skips that trace entirely.

        Returns:
            A `KerasTensor` with shape `(1, None, hidden_dim)`, matching
            this encoder's always-batch-1, variable-token-count output.
        """
        return keras.KerasTensor(
            shape=(1, None, self.hidden_dim),
            dtype=self.compute_dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "intermediate_dim": self.intermediate_dim,
                "rope_theta": self.rope_theta,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "activation": self.activation,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config


class Mistral3PatchMerger(keras.layers.Layer):
    """Spatially merge Pixtral patches for Mistral3.

    Every 2x2 group of vision patches is concatenated along the feature
    dimension and projected from hidden_dim * 4 back to hidden_dim.

    For the Mistral3 configuration:

        1024 * 2 * 2 = 4096
        4096 -> 1024
    """

    def __init__(
        self,
        hidden_dim=1024,
        spatial_merge_size=MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE,
        patch_size=14,
        image_size=1540,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size

        self.merging_layer = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            name="merging_layer",
        )

    def _merge_image(
        self,
        image_features,
        height,
        width,
    ):
        """Merge one image's patch sequence spatially."""

        merge_size = self.spatial_merge_size
        hidden_dim = self.hidden_dim

        # [tokens, hidden_dim]
        image_features = ops.reshape(
            image_features,
            (
                height,
                width,
                hidden_dim,
            ),
        )

        # Convert:
        #
        # [H, W, D]
        #
        # -> [H/2, 2, W/2, 2, D]
        #
        # The two dimensions of size `merge_size` represent the
        # spatial neighborhood being merged.
        merged_height = height // merge_size
        merged_width = width // merge_size

        image_features = ops.reshape(
            image_features,
            (
                merged_height,
                merge_size,
                merged_width,
                merge_size,
                hidden_dim,
            ),
        )

        # [H/2, 2, W/2, 2, D]
        #
        # -> [H/2, W/2, D, 2, 2]
        #
        # This matches the channel-major ordering produced by
        # PyTorch F.unfold used by HF:
        #
        # [D * merge_size * merge_size, num_windows]
        image_features = ops.transpose(
            image_features,
            (0, 2, 4, 1, 3),
        )

        image_features = ops.reshape(
            image_features,
            (
                merged_height * merged_width,
                hidden_dim * merge_size * merge_size,
            ),
        )

        return image_features

    def call(
        self,
        image_features,
        image_sizes,
        max_patch_height,
        max_patch_width,
    ):
        """Merge valid spatial neighborhoods using static-shape indexing.

        `image_features` is a padded canvas: the exact-count, per-image
        concatenated patch features come first (in the same order/offsets
        as `image_sizes`), followed by zero rows padding the row count up
        to `num_images * max_patch_height * max_patch_width`. That total is
        a function of *shapes* only (never of the values inside
        `image_sizes`), which is what keeps this layer compatible with
        `jax.jit` tracing — unlike a `nonzero`-based compaction, whose
        output shape depends on how many merge windows are actually valid.

        `max_patch_height`/`max_patch_width` must be passed in (rather than
        derived from `image_sizes` via `ops.max`) so that, when they are
        plain Python ints, the output buffer size is a trace-time constant.

        Returns:
            A tuple `(merged_padded, valid_count)`: `merged_padded` has
            shape `(num_images * (max_patch_height // spatial_merge_size) *
            (max_patch_width // spatial_merge_size), hidden_dim)`, with real
            merged windows as a prefix and the remainder as unused padding;
            `valid_count` is the scalar number of real windows in that
            prefix.
        """
        image_features = ops.reshape(
            image_features,
            (-1, ops.shape(image_features)[-1]),
        )
        image_sizes = ops.cast(
            ops.convert_to_tensor(image_sizes),
            "int32",
        )
        patch_heights = image_sizes[:, 0] // self.patch_size
        patch_widths = image_sizes[:, 1] // self.patch_size
        patch_counts = patch_heights * patch_widths
        num_images = ops.shape(image_sizes)[0]
        num_tokens = ops.shape(image_features)[0]
        merge_size = self.spatial_merge_size

        image_offsets = ops.cumsum(patch_counts) - patch_counts
        cumulative_ends = ops.cumsum(patch_counts)
        total_valid_tokens = ops.sum(patch_counts)

        token_indices = ops.arange(num_tokens, dtype="int32")
        # Static-shape stand-in for `ops.repeat(arange(num_images),
        # patch_counts)`: for each token, count how many images' ranges end
        # at or before it, giving that token's image id.
        image_ids = ops.sum(
            ops.cast(
                ops.expand_dims(token_indices, 1)
                >= ops.expand_dims(cumulative_ends, 0),
                "int32",
            ),
            axis=1,
        )
        image_ids = ops.clip(image_ids, 0, num_images - 1)

        widths_per_token = ops.take(patch_widths, image_ids, axis=0)
        heights_per_token = ops.take(patch_heights, image_ids, axis=0)
        offsets_per_token = ops.take(image_offsets, image_ids, axis=0)
        local_indices = token_indices - offsets_per_token
        local_rows = local_indices // widths_per_token
        local_columns = local_indices % widths_per_token

        is_real = token_indices < total_valid_tokens
        valid_windows = ops.logical_and(
            is_real,
            ops.logical_and(
                local_rows % merge_size == 0,
                local_columns % merge_size == 0,
            ),
        )
        valid_windows = ops.logical_and(
            valid_windows,
            local_rows + merge_size <= heights_per_token,
        )
        valid_windows = ops.logical_and(
            valid_windows,
            local_columns + merge_size <= widths_per_token,
        )

        # `offsets_per_token + local_rows * widths_per_token +
        # local_columns` is algebraically just `token_indices` (floor-div/
        # mod recombine exactly), so the window's top-left patch is the
        # token itself. The window covers `merge_size` rows and
        # `merge_size` columns of patches, in row-major order, matching the
        # channel ordering `_merge_image` produces.
        max_index = num_tokens - 1
        patch_indices = ops.stack(
            [
                ops.clip(
                    token_indices + row * widths_per_token + col,
                    0,
                    max_index,
                )
                for row in range(merge_size)
                for col in range(merge_size)
            ],
            axis=1,
        )
        patches = ops.take(image_features, patch_indices, axis=0)
        patches = ops.transpose(patches, (0, 2, 1))
        patches = ops.reshape(
            patches,
            (-1, self.hidden_dim * merge_size * merge_size),
        )
        merged_all = self.merging_layer(patches)

        valid_int = ops.cast(valid_windows, "int32")
        rank = ops.cumsum(valid_int) - valid_int
        capacity = (
            num_images
            * (max_patch_height // merge_size)
            * (max_patch_width // merge_size)
        )
        scatter_target = ops.where(valid_windows, rank, capacity)
        buffer = ops.zeros((capacity + 1, self.hidden_dim), merged_all.dtype)
        buffer = ops.scatter_update(
            buffer,
            ops.expand_dims(scatter_target, axis=1),
            merged_all,
        )
        merged_padded = buffer[:capacity]
        valid_count = ops.sum(valid_int)
        return merged_padded, valid_count

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
            }
        )
        return config


class Mistral3MultiModalProjector(keras.layers.Layer):
    """Multimodal projector used by Mistral3.

    Vision features are normalized, spatially merged, and projected into
    the Mistral text-model hidden dimension.
    """

    def __init__(
        self,
        vision_hidden_dim=1024,
        text_hidden_dim=5120,
        spatial_merge_size=MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE,
        patch_size=14,
        layer_norm_epsilon=1e-6,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
        image_size=1540,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vision_hidden_dim = vision_hidden_dim
        self.image_size = image_size
        self.text_hidden_dim = text_hidden_dim
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.projector_hidden_act = projector_hidden_act
        self.multimodal_projector_bias = multimodal_projector_bias

        self.norm = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            name="norm",
        )

        self.patch_merger = Mistral3PatchMerger(
            hidden_dim=vision_hidden_dim,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            image_size=image_size,
            name="patch_merger",
        )

        self.linear_1 = keras.layers.Dense(
            text_hidden_dim,
            use_bias=multimodal_projector_bias,
            kernel_initializer=_mistral_kernel_initializer(),
            name="linear_1",
        )

        self.act = keras.activations.get(
            projector_hidden_act,
        )

        self.linear_2 = keras.layers.Dense(
            text_hidden_dim,
            use_bias=multimodal_projector_bias,
            kernel_initializer=_mistral_kernel_initializer(),
            name="linear_2",
        )

    def call(
        self,
        image_features,
        image_sizes,
        max_patch_height,
        max_patch_width,
    ):
        image_features = self.norm(
            image_features,
        )

        image_features, valid_count = self.patch_merger(
            image_features,
            image_sizes=image_sizes,
            max_patch_height=max_patch_height,
            max_patch_width=max_patch_width,
        )

        hidden_states = self.linear_1(
            image_features,
        )

        hidden_states = self.act(
            hidden_states,
        )

        hidden_states = self.linear_2(
            hidden_states,
        )

        return hidden_states[:valid_count]

    def compute_output_spec(
        self,
        image_features,
        image_sizes,
        max_patch_height,
        max_patch_width,
    ):
        """Return the output shape spec without running the forward pass.

        Needed for the same reason as
        `Mistral3VisionEncoder.compute_output_spec`: `MistralBackbone`
        wraps this projector inside a Keras functional model, and its
        output row count (`valid_count`, the number of real patch-merge
        windows) is data-dependent. Tracing `call()` on the JAX backend to
        infer that shape hits the same `InconclusiveDimensionOperation`
        issue, since it relies on `Mistral3PatchMerger`'s
        `ops.nonzero`-equivalent cumulative-sum indexing over symbolic
        dimensions.

        Returns:
            A `KerasTensor` with shape `(None, text_hidden_dim)`.
        """
        return keras.KerasTensor(
            shape=(None, self.text_hidden_dim),
            dtype=self.compute_dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_hidden_dim": self.vision_hidden_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "projector_hidden_act": self.projector_hidden_act,
                "multimodal_projector_bias": (self.multimodal_projector_bias),
            }
        )
        return config
