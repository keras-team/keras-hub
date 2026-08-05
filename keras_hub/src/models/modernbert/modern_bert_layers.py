import keras
from keras import layers
from keras import ops


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertMLP(layers.Layer):
    """ModernBERT MLP block using Gated Linear Units (GeGLU).

    Implements the mathematical operation:
    `output = wo(activation(wi_0(x)) * wi_1(x))`

    Args:
        hidden_dim: int. The input and output dimensionality of the layer.
        intermediate_dim: int. The inner gated projection dimensionality.
        activation: string or callable. The activation function configuration
            to apply to the gating projection. Defaults to `gelu_approximate`.
        dtype: string or `keras.DTypePolicy`. The precision policy used for the
            layer's computations and weights. Defaults to `None`.

    Examples:
    ```python
    from keras import ops
    import numpy as np

    mlp = ModernBertMLP(hidden_dim=256, intermediate_dim=512)
    inputs = ops.convert_to_tensor(np.random.normal(size=(2, 16, 256)))
    outputs = mlp(inputs)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation="gelu",
        dtype=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = keras.config.dtype_policy()

        super().__init__(dtype=dtype, **kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = keras.activations.get(activation)

        self.wi_0 = layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=dtype,
            name="wi_0",
        )

        self.wi_1 = layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=dtype,
            name="wi_1",
        )

        self.wo = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="wo",
        )

    def build(self, input_shape):
        self.wi_0.build(input_shape)
        self.wi_1.build(input_shape)

        self.wo.build((*input_shape[:-1], self.intermediate_dim))

        super().build(input_shape)

    def call(self, x):
        """Forward pass of the GeGLU MLP layer."""
        gate = self.activation(self.wi_0(x))
        value = self.wi_1(x)

        return self.wo(gate * value)

    def get_config(self):
        """Returns the serialization configuration of the MLP layer."""
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertAttention(layers.Layer):
    """ModernBERT attention layer.
    This layer implements multi-head self-attention. It optionally supports
    both Rotary Position Embeddings (RoPE) and sequence
    local sliding-window masks to optimize computation over longer contexts.

    Args:
        hidden_dim: int. The size of the input transformer states.
        num_heads: int. The number of self-attention heads.
        rotary_embedding: `keras.layers.Layer` or callable. An instance of a
        rotary position embedding layer to position-encode
        query and key tensors. Defaults to `None`.
        local_attention_window: int. Window limit for local
        sliding-window attention.
            If `None`, global attention is executed. Defaults to `None`.
        dropout: float. Attention dropout score probability. Defaults to `0.0`.
        dtype: string or `keras.DTypePolicy`. The precision policy used for the
            layer's computations and weights.

    Raises:
        ValueError: If `hidden_dim` is not perfectly divisible by `num_heads`.

    Examples:
    ```python
    from keras import ops
    import numpy as np

    attention = ModernBertAttention(hidden_dim=256, num_heads=4)
    inputs = ops.convert_to_tensor(np.random.normal(size=(2, 32, 256)))
    outputs = attention(inputs)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        rotary_embedding=None,
        local_attention_window=None,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = keras.config.dtype_policy()

        super().__init__(dtype=dtype, **kwargs)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` ({hidden_dim}) must be divisible "
                f"by `num_heads` ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rotary_embedding = rotary_embedding
        self.local_attention_window = local_attention_window
        self.dropout = dropout

        self.qkv = layers.Dense(
            3 * hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="qkv",
        )

        self.output_dense = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="output_dense",
        )

        self.attn_dropout = layers.Dropout(
            dropout,
            dtype=dtype,
            name="attention_dropout",
        )

    def build(self, input_shape):
        self.qkv.build(input_shape)

        self.output_dense.build(
            (
                input_shape[0],
                input_shape[1],
                self.hidden_dim,
            )
        )

        super().build(input_shape)

    def _get_sliding_window_mask(self, seq_len, dtype):
        """Return the bidirectional local-attention mask."""

        half_window = self.local_attention_window // 2

        positions = ops.arange(seq_len)

        distance = ops.abs(positions[:, None] - positions[None, :])

        mask = distance <= half_window

        return ops.cast(mask, dtype)

    def _apply_rope(self, x):
        """Apply RotaryEmbedding while preserving [B, H, T, D]."""

        batch_size = ops.shape(x)[0]
        num_heads = ops.shape(x)[1]
        seq_len = ops.shape(x)[2]

        x = ops.reshape(
            x,
            (
                batch_size * num_heads,
                seq_len,
                self.head_dim,
            ),
        )

        x = self.rotary_embedding(x)

        x = ops.reshape(
            x,
            (
                batch_size,
                num_heads,
                seq_len,
                self.head_dim,
            ),
        )

        return x

    def call(
        self,
        x,
        padding_mask=None,
        training=None,
    ):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # QKV projection
        qkv = self.qkv(x)

        q, k, v = ops.split(
            qkv,
            3,
            axis=-1,
        )

        # [B, T, H*D] -> [B, T, H, D]
        q = ops.reshape(
            q,
            (
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ),
        )

        k = ops.reshape(
            k,
            (
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ),
        )

        v = ops.reshape(
            v,
            (
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ),
        )

        # [B, T, H, D] -> [B, H, T, D]
        q = ops.transpose(
            q,
            (0, 2, 1, 3),
        )

        k = ops.transpose(
            k,
            (0, 2, 1, 3),
        )

        v = ops.transpose(
            v,
            (0, 2, 1, 3),
        )

        # Rotary position embedding
        if self.rotary_embedding is not None:
            q = self._apply_rope(q)
            k = self._apply_rope(k)

        # Attention scores
        scale = self.head_dim**-0.5

        scores = ops.matmul(
            q,
            ops.transpose(
                k,
                (0, 1, 3, 2),
            ),
        )

        scores = scores * scale

        # Local sliding-window attention
        if self.local_attention_window is not None:
            local_mask = self._get_sliding_window_mask(
                seq_len,
                scores.dtype,
            )

            local_mask = local_mask[None, None, :, :]

            scores = scores + (1.0 - local_mask) * ops.cast(
                -1e9,
                scores.dtype,
            )

        # Padding mask
        if padding_mask is not None:
            padding_mask = ops.cast(
                padding_mask,
                scores.dtype,
            )

            padding_mask = padding_mask[:, None, None, :]

            scores = scores + (1.0 - padding_mask) * ops.cast(
                -1e9,
                scores.dtype,
            )

        # Softmax
        probabilities = ops.softmax(
            ops.cast(
                scores,
                "float32",
            ),
            axis=-1,
        )

        probabilities = ops.cast(
            probabilities,
            v.dtype,
        )

        probabilities = self.attn_dropout(
            probabilities,
            training=training,
        )

        # Attention output
        output = ops.matmul(
            probabilities,
            v,
        )

        # [B, H, T, D] -> [B, T, H, D]
        output = ops.transpose(
            output,
            (0, 2, 1, 3),
        )

        # [B, T, H, D] -> [B, T, hidden_dim]
        output = ops.reshape(
            output,
            (
                batch_size,
                seq_len,
                self.hidden_dim,
            ),
        )

        # Output projection
        output = self.output_dense(output)

        return output

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "local_attention_window": (self.local_attention_window),
                "dropout": self.dropout,
                "rotary_embedding": (
                    keras.saving.serialize_keras_object(self.rotary_embedding)
                    if self.rotary_embedding is not None
                    else None
                ),
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        rotary_config = config.get("rotary_embedding")

        if rotary_config is not None:
            config["rotary_embedding"] = keras.saving.deserialize_keras_object(
                rotary_config
            )

        return cls(**config)


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertEncoderLayer(layers.Layer):
    """ModernBERT encoder block.

    This layer encapsulates an instance of `ModernBertAttention`, followed by a
    `ModernBertMLP` block. Residual connections use pre-layer normalization with
    `keras.layers.LayerNormalization`.

    Args:
        hidden_dim: int. The hidden state dimension of the block.
        intermediate_dim: int. Gated linear unit intermediate projection
        dimension.
        num_heads: int. The number of self-attention heads.
        rotary_embedding: `keras.layers.Layer` or callable. An instance of a
        rotary position embedding layer passed to the underlying
        attention object.
            Defaults to `None`.
        local_attention_window: int. Maximum sequence length context window
        limit for sliding window operations. Set to `None` for global attention.
            Defaults to `None`.
        dropout: float. Attention map and feature output dropout probability.
            Defaults to `0.0`.
        layer_norm_epsilon: float. Small value applied inside the
        `RMSNormalization` layers to avoid zero division.
            Defaults to `1e-5`.
        dtype: string or `keras.DTypePolicy`. The precision policy used for the
            layer's computations and weights. Defaults to `None`.

    Examples:
    ```python
    from keras import ops
    import numpy as np

    encoder = ModernBertEncoderLayer(
        hidden_dim=256,
        intermediate_dim=512,
        num_heads=4,
    )
    inputs = ops.convert_to_tensor(np.random.normal(size=(2, 32, 256)))
    outputs = encoder(inputs)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        layer_idx,
        rotary_embedding=None,
        local_attention_window=None,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        if dtype is None:
            dtype = keras.config.dtype_policy()

        super().__init__(dtype=dtype, **kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.rotary_embedding = rotary_embedding
        self.local_attention_window = local_attention_window
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        # ModernBERT layer 0 has no attention LayerNorm.
        if layer_idx == 0:
            self.attn_norm = layers.Identity(
                name="attention_norm",
                dtype=dtype,
            )
        else:
            self.attn_norm = layers.LayerNormalization(
                epsilon=layer_norm_epsilon,
                center=False,
                scale=True,
                dtype=dtype,
                name="attention_norm",
            )

        self.attn = ModernBertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            rotary_embedding=rotary_embedding,
            local_attention_window=local_attention_window,
            dropout=dropout,
            dtype=dtype,
            name="attention",
        )

        self.mlp_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            center=False,
            scale=True,
            dtype=dtype,
            name="mlp_norm",
        )

        self.mlp = ModernBertMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dtype=dtype,
            name="mlp",
        )

        self.attn_dropout = layers.Dropout(
            dropout,
            dtype=dtype,
        )

        self.mlp_dropout = layers.Dropout(
            dropout,
            dtype=dtype,
        )

    def build(self, input_shape):
        self.attn_norm.build(input_shape)
        self.attn.build(input_shape)
        self.mlp_norm.build(input_shape)
        self.mlp.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        x,
        padding_mask=None,
        training=None,
    ):
        """Forward pass of the complete encoder layer block."""

        # Attention residual block
        residual = x
        x = self.attn_norm(x)
        x = self.attn(
            x,
            padding_mask=padding_mask,
            training=training,
        )

        x = self.attn_dropout(
            x,
            training=training,
        )
        if x.dtype != residual.dtype:
            x = ops.cast(
                x,
                residual.dtype,
            )
        x = residual + x

        # MLP residual block
        residual = x

        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_dropout(
            x,
            training=training,
        )
        if x.dtype != residual.dtype:
            x = ops.cast(
                x,
                residual.dtype,
            )
        x = residual + x

        return x

    def get_config(self):
        """Returns the serialization configuration of the encoder layer."""
        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "layer_idx": self.layer_idx,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "rotary_embedding": (
                    keras.saving.serialize_keras_object(self.rotary_embedding)
                    if self.rotary_embedding is not None
                    else None
                ),
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        if config.get("rotary_embedding") is not None:
            config["rotary_embedding"] = keras.saving.deserialize_keras_object(
                config["rotary_embedding"]
            )

        return super().from_config(config)
