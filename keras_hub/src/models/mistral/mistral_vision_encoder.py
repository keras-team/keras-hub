import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export

MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE = 2


def _mistral_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


class Mistral3ImageFeatureExtractor(keras.layers.Layer):
    """Computes projected Mistral3 image features.

    Wrapped in a `Layer` so shape queries see real tensors, not
    graph-construction-time placeholders.

    Args:
        vision_encoder: A `Mistral3VisionEncoder` instance.
        multimodal_projector: A `Mistral3MultiModalProjector` instance.
        vision_feature_layer: int. Which vision encoder hidden state to
            project; only `-1` (the final hidden state) is supported.
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
        # `vision_encoder`'s padded-canvas output already matches what
        # `Mistral3MultiModalProjector` expects; no further padding needed.
        image_features = ops.squeeze(image_features, axis=0)

        patch_size = self.multimodal_projector.patch_size
        max_patch_height = ops.shape(pixel_values)[2] // patch_size
        max_patch_width = ops.shape(pixel_values)[3] // patch_size

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

    Finding placeholder positions is a data-dependent (`nonzero`-style) op,
    incompatible with `jax.jit` tracing. Run this eagerly with NumPy in
    preprocessing, before `token_ids` reaches the model; the result is
    passed in as a plain tensor input, so the model itself only needs a
    static-shape `scatter_update`.

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


def compute_resize_size(height, width, longest_edge, patch_size):
    """Computes the resize target for one image.

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

        # `placeholder_indices` may arrive flat or batched `(batch, N)` —
        # flatten to match `flat_embeddings`.
        if len(ops.shape(placeholder_indices)) == 2:
            placeholder_indices = ops.reshape(placeholder_indices, (-1,))
        # Drop `image_features`' trailing zero padding (from
        # `Mistral3MultiModalProjector`'s fixed-capacity output) down to the
        # real prefix that fills every placeholder.
        num_placeholders = ops.shape(placeholder_indices)[0]
        image_features = image_features[:num_placeholders]
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
    """2D rotary positional embedding for the Mistral3 vision encoder.

    Frequencies are built from 2D patch coordinates: the first half of the
    frequency dims encodes height, the second half width.
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

    def _compute_inv_freq(self):
        max_patches = self.max_patches_per_side

        freq_indices = ops.arange(0, self.head_dim, 2, dtype="float32")
        freqs = ops.divide(
            1.0,
            ops.power(self.rope_theta, freq_indices / self.head_dim),
        )

        height_indices = ops.arange(max_patches, dtype="float32")
        width_indices = ops.arange(max_patches, dtype="float32")

        freqs_h = ops.einsum(
            "i,j->ij",
            height_indices,
            freqs[::2],
        )

        freqs_w = ops.einsum(
            "i,j->ij",
            width_indices,
            freqs[1::2],
        )

        half_dim = self.head_dim // 4

        # [H, 1, D/4] -> [H, W, D/4]
        freqs_h = ops.broadcast_to(
            ops.expand_dims(freqs_h, axis=1),
            (max_patches, max_patches, half_dim),
        )

        # [1, W, D/4] -> [H, W, D/4]
        freqs_w = ops.broadcast_to(
            ops.expand_dims(freqs_w, axis=0),
            (max_patches, max_patches, half_dim),
        )

        inv_freq = ops.concatenate(
            [freqs_h, freqs_w],
            axis=-1,
        )

        inv_freq = ops.reshape(
            inv_freq,
            (-1, self.head_dim // 2),
        )

        return ops.concatenate(
            [inv_freq, inv_freq],
            axis=-1,
        )

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
            self._compute_inv_freq(),
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
    """Multi-head self-attention used by the Mistral3 vision encoder.

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
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.o_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            dtype=self.dtype_policy,
            name="o_proj",
        )

        self.attention_dropout = keras.layers.Dropout(
            dropout, dtype=self.dtype_policy
        )

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
    """SwiGLU MLP used by the Mistral3 vision encoder.

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
            dtype=self.dtype_policy,
            name="gate_proj",
        )

        self.up_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            dtype=self.dtype_policy,
            name="up_proj",
        )

        self.down_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=_mistral_kernel_initializer(),
            dtype=self.dtype_policy,
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
    """One Mistral3 vision transformer encoder layer.

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
            dtype=self.dtype_policy,
            name="attention_norm",
        )

        self.attention = Mistral3VisionAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            dtype=self.dtype_policy,
            name="attention",
        )

        self.ffn_norm = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="ffn_norm",
        )

        self.feed_forward = Mistral3VisionMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation=activation,
            dtype=self.dtype_policy,
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
    """Vision encoder used by Mistral3.

    This is not exposed as a standalone model. It is the vision tower
    consumed by the Mistral3 multimodal architecture.

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
            dtype=self.dtype_policy,
            name="patch_conv",
        )

        self.ln_pre = keras.layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="ln_pre",
        )

        self.patch_positional_embedding = Mistral3VisionRotaryEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            dtype=self.dtype_policy,
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
                    dtype=self.dtype_policy,
                    name=f"transformer_layer_{i}",
                )
            )

    def _create_position_ids_for_images(
        self,
        image_sizes,
        max_patch_height,
        max_patch_width,
    ):
        """Create mesh-grid position IDs matching the compacted layout.

        Mirrors `_extract_patch_sequences`'s compaction (see that method for
        why a cumulative-sum rank is used instead of `ops.nonzero`) so
        position IDs line up with the patches they belong to.
        """
        image_sizes = ops.cast(image_sizes, "int32")
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

        capacity = num_images * max_patch_height * max_patch_width
        valid_int = ops.cast(valid_patches, "int32")
        rank = ops.cumsum(valid_int) - valid_int
        scatter_target = ops.where(valid_patches, rank, capacity)
        buffer = ops.zeros((capacity + 1,), dtype="int32")
        buffer = ops.scatter_update(
            buffer,
            ops.expand_dims(scatter_target, axis=1),
            position_grid,
        )
        return buffer[:capacity]

    def _create_block_attention_mask(self, patch_counts, dtype, capacity):
        """Create a block-diagonal mask over the padded, compacted layout.

        `patch_counts` gives each image's real patch count; slots at or
        past `sum(patch_counts)` are the tail padding produced by
        `_extract_patch_sequences` and are assigned a sentinel block ID
        (`num_images`) so they never attend to, or are attended to by, a
        real patch. Computes each slot's image ID via a cumulative-sum
        comparison rather than `ops.repeat(..., patch_counts)` (see
        `_extract_patch_sequences` for why).
        """
        patch_counts = ops.cast(patch_counts, "int32")
        num_images = ops.shape(patch_counts)[0]
        total_valid = ops.sum(patch_counts)
        cumulative_ends = ops.cumsum(patch_counts)

        token_indices = ops.arange(capacity, dtype="int32")
        image_ids = ops.sum(
            ops.cast(
                ops.expand_dims(token_indices, axis=1)
                >= ops.expand_dims(cumulative_ends, axis=0),
                "int32",
            ),
            axis=1,
        )
        image_ids = ops.where(
            token_indices >= total_valid,
            num_images,
            image_ids,
        )

        same_block = ops.equal(
            ops.expand_dims(image_ids, axis=0),
            ops.expand_dims(image_ids, axis=1),
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
        """Compact per-image patches into cumsum-offset order, tail-padded.

        Output has the same row count as the input
        (`num_images * max_patch_height * max_patch_width`): valid patches
        move to the front, in per-image row-major order (the layout
        `Mistral3PatchMerger` expects), remainder zeroed. Uses a
        cumulative-sum rank instead of `ops.nonzero`, since `nonzero`'s
        output shape depends on tensor values and breaks `jax.jit` tracing.
        """
        image_sizes = ops.cast(image_sizes, "int32")
        patch_heights = image_sizes[:, 0] // self.patch_size
        patch_widths = image_sizes[:, 1] // self.patch_size
        num_images = ops.shape(patch_embeds)[0]
        max_patch_height = ops.shape(patch_embeds)[1]
        max_patch_width = ops.shape(patch_embeds)[2]
        hidden_dim = ops.shape(patch_embeds)[-1]

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
        patch_embeds = ops.reshape(patch_embeds, (-1, hidden_dim))
        valid_patches = ops.reshape(valid_patches, (-1,))

        capacity = num_images * max_patch_height * max_patch_width
        valid_int = ops.cast(valid_patches, "int32")
        rank = ops.cumsum(valid_int) - valid_int
        scatter_target = ops.where(valid_patches, rank, capacity)
        buffer = ops.zeros((capacity + 1, hidden_dim), patch_embeds.dtype)
        buffer = ops.scatter_update(
            buffer,
            ops.expand_dims(scatter_target, axis=1),
            patch_embeds,
        )
        return buffer[:capacity]

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
        capacity = (
            ops.shape(pixel_values)[0] * max_patch_height * max_patch_width
        )
        attention_mask = self._create_block_attention_mask(
            patch_counts,
            patch_embeds.dtype,
            capacity,
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
        """Declare the output shape without tracing `call()`.

        `MistralBackbone` builds this encoder into a functional model with
        a dynamic `pixel_values` shape; tracing `call()`'s data-dependent
        shape ops on symbolic dimensions isn't supported by the JAX
        backend's graph tracer.

        Returns:
            A `KerasTensor` with shape `(1, None, hidden_dim)`.
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
    """Spatially merge vision patches for Mistral3.

    Every `spatial_merge_size x spatial_merge_size` group of vision patches
    is concatenated along the feature dimension and projected back down to
    `hidden_dim`.
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
            dtype=self.dtype_policy,
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

        # [tokens, hidden_dim] -> [H, W, D]
        image_features = ops.reshape(
            image_features,
            (
                height,
                width,
                hidden_dim,
            ),
        )

        # [H, W, D] -> [H/2, 2, W/2, 2, D]: split each spatial dim into
        # (windows, merge_size).
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

        # [H/2, 2, W/2, 2, D] -> [H/2, W/2, D, 2, 2], matching the
        # channel-major ordering of PyTorch's F.unfold (used by HF).
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

        `image_features` is a padded canvas, sized
        `num_images * max_patch_height * max_patch_width` (a function of
        shapes only, `jax.jit`-safe): real per-image patches first, zero
        rows after. `max_patch_height`/`max_patch_width` are passed in
        rather than derived via `ops.max` so the output buffer size stays a
        trace-time constant.

        Returns:
            A tuple `(merged_padded, valid_count)`: `merged_padded` has
            real merged windows as a prefix, zero-padded after; `valid_count`
            is the scalar number of real windows in that prefix.
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
        # patch_counts)`.
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

        # The token itself is the window's top-left patch (floor-div/mod
        # recombine exactly). Gather the `merge_size x merge_size` window
        # in row-major order, matching `_merge_image`'s channel ordering.
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
        layer_norm_epsilon=1e-5,
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
            dtype=self.dtype_policy,
            name="norm",
        )

        self.patch_merger = Mistral3PatchMerger(
            hidden_dim=vision_hidden_dim,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            image_size=image_size,
            dtype=self.dtype_policy,
            name="patch_merger",
        )

        self.linear_1 = keras.layers.Dense(
            text_hidden_dim,
            use_bias=multimodal_projector_bias,
            kernel_initializer=_mistral_kernel_initializer(),
            dtype=self.dtype_policy,
            name="linear_1",
        )

        self.act = keras.activations.get(
            projector_hidden_act,
        )

        self.linear_2 = keras.layers.Dense(
            text_hidden_dim,
            use_bias=multimodal_projector_bias,
            kernel_initializer=_mistral_kernel_initializer(),
            dtype=self.dtype_policy,
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

        # `valid_count` is unused: trimming to it needs a data-dependent
        # slice, which breaks `jax.jit` tracing. The padded output is
        # trimmed downstream instead, in `Mistral3ImageTextEmbeddingMerger`.
        image_features, _ = self.patch_merger(
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

        return hidden_states

    def compute_output_spec(
        self,
        image_features,
        image_sizes,
        max_patch_height,
        max_patch_width,
    ):
        """Declare the output shape without tracing `call()`.

        Same reason as `Mistral3VisionEncoder.compute_output_spec`.

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
