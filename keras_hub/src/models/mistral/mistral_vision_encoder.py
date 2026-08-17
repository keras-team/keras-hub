import keras
from keras import ops


def _mistral_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


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

    def build(self, input_shape):
        max_patches = self.max_patches_per_side

        # HF:
        #
        # freqs = 1 / (
        #     base ** (arange(0, dim, 2) / dim)
        # )
        #
        # freqs_h = outer(h, freqs[::2])
        # freqs_w = outer(w, freqs[1::2])
        #
        # Result has shape:
        # [max_patches, max_patches, head_dim // 2]
        #
        # Then it is flattened and duplicated to head_dim.

        freq_indices = ops.arange(0, self.head_dim, 2, dtype="float32")
        freqs = ops.power(
            self.rope_theta,
            -freq_indices / self.head_dim,
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

        # [H, 1, D/4] -> [H, W, D/4]
        freqs_h = ops.broadcast_to(
            ops.expand_dims(freqs_h, axis=1),
            (max_patches, max_patches, ops.shape(freqs_h)[-1]),
        )

        # [1, W, D/4] -> [H, W, D/4]
        freqs_w = ops.broadcast_to(
            ops.expand_dims(freqs_w, axis=0),
            (max_patches, max_patches, ops.shape(freqs_w)[-1]),
        )

        inv_freq = ops.concatenate(
            [freqs_h, freqs_w],
            axis=-1,
        )

        inv_freq = ops.reshape(
            inv_freq,
            (-1, self.head_dim // 2),
        )

        # HF duplicates the frequencies:
        #
        # inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        inv_freq = ops.concatenate(
            [inv_freq, inv_freq],
            axis=-1,
        )

        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(max_patches * max_patches, self.head_dim),
            initializer=keras.initializers.Constant(
                ops.convert_to_numpy(inv_freq)
            ),
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
        # position_ids:
        # [batch, sequence]
        #
        # inv_freq[position_ids]
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
    # q/k:
    # [batch, heads, sequence, head_dim]
    #
    # cos/sin:
    # [batch, sequence, head_dim]
    #
    # Unlike HF (whose pixtral "batch" dim is always 1, so unsqueeze_dim=0
    # works), this port keeps the real batch dimension, so we broadcast
    # over the heads axis (axis=1) instead.

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


class Mistral3VisionEncoder(keras.layers.Layer):
    """
    Pixtral vision encoder used by Mistral3.

    This is not exposed as a standalone Pixtral model. It is the vision
    tower consumed by the Mistral3 multimodal architecture.
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

        # HF uses Conv2D with:
        # in_channels=3
        # out_channels=1024
        # kernel_size=14
        # stride=14
        # bias=False
        #
        # Keras defaults to channels_last, so call() converts the HF-style
        # [B, C, H, W] input before this layer.
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

    def _create_position_ids(self, height, width):
        grid_height = ops.arange(
            height,
            dtype="int32",
        )

        grid_width = ops.arange(
            width,
            dtype="int32",
        )

        h_grid = ops.broadcast_to(
            ops.expand_dims(grid_height, axis=1),
            (height, width),
        )

        w_grid = ops.broadcast_to(
            ops.expand_dims(grid_width, axis=0),
            (height, width),
        )

        max_width = self.image_size // self.patch_size

        position_ids = h_grid * max_width + w_grid

        return ops.reshape(
            position_ids,
            (1, height * width),
        )

    def _create_block_attention_mask(self, sequence_length):
        # For the initial implementation this represents one image,
        # therefore every patch can attend to every other patch.
        #
        # Multi-image block-diagonal masking will be added once
        # single-image numerical parity is established.
        return None

    def call(
        self,
        pixel_values,
        image_sizes=None,
        training=None,
    ):
        # HF input layout:
        # [batch, channels, height, width]
        #
        # Keras Conv2D:
        # [batch, height, width, channels]
        pixel_values = ops.transpose(
            pixel_values,
            (0, 2, 3, 1),
        )

        patch_embeds = self.patch_conv(pixel_values)

        batch_size = ops.shape(patch_embeds)[0]
        height = ops.shape(patch_embeds)[1]
        width = ops.shape(patch_embeds)[2]

        # Initial implementation: one image per batch element and all images
        # share the resulting patch grid. We will add image_sizes-based
        # cropping/concatenation after single-image parity.
        patch_embeds = ops.reshape(
            patch_embeds,
            (
                batch_size,
                height * width,
                self.hidden_dim,
            ),
        )

        patch_embeds = self.ln_pre(patch_embeds)

        position_ids = self._create_position_ids(
            height,
            width,
        )

        # Broadcast position ids to batch.
        position_ids = ops.broadcast_to(
            position_ids,
            (
                batch_size,
                height * width,
            ),
        )

        cos, sin = self.patch_positional_embedding(
            position_ids,
            dtype=patch_embeds.dtype,
        )

        hidden_states = patch_embeds

        attention_mask = self._create_block_attention_mask(
            height * width,
        )

        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                training=training,
            )

        return hidden_states

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
