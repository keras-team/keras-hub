import keras
from keras import layers
from keras import ops

from keras_hub.src.utils.keras_utils import gelu_approximate


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
        activation=gelu_approximate,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = keras.activations.get(activation)

        self.wi_0 = layers.Dense(
            intermediate_dim, use_bias=False, dtype=dtype, name="wi_0"
        )
        self.wi_1 = layers.Dense(
            intermediate_dim, use_bias=False, dtype=dtype, name="wi_1"
        )
        self.wo = layers.Dense(
            hidden_dim, use_bias=False, dtype=dtype, name="wo"
        )

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
        super().__init__(dtype=dtype, **kwargs)

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.rotary_embedding = rotary_embedding
        self.local_attention_window = local_attention_window
        self.dropout = dropout

        self.qkv = layers.Dense(
            hidden_dim * 3, use_bias=False, dtype=dtype, name="qkv"
        )
        self.output_dense = layers.Dense(
            hidden_dim, use_bias=False, dtype=dtype, name="output_dense"
        )

        self.attn_dropout = layers.Dropout(dropout, dtype=dtype)

    def _get_sliding_window_mask(self, seq_len, dtype):
        """
        Generates a binary sequence mask for local
        sliding-window attention.

        This function creates a boolean square matrix representing
        whether tokens fall within the context boundaries of a
        centralized attention token window.
        """
        idx = ops.arange(seq_len)
        dist = ops.abs(idx[:, None] - idx[None, :])
        max_distance = self.local_attention_window // 2
        return ops.cast(dist <= max_distance, dtype)

    def call(self, x, padding_mask=None, training=None):
        """Forward pass of the multi-head self-attention layer."""
        b = ops.shape(x)[0]
        t = ops.shape(x)[1]

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, t, 3, self.num_heads, self.head_dim))
        q, k, v = ops.unstack(qkv, axis=2)

        # rotary
        if self.rotary_embedding is not None:
            q = self.rotary_embedding(q)
            k = self.rotary_embedding(k)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 3, 1))
        v = ops.transpose(v, (0, 2, 1, 3))

        scale = self.head_dim**-0.5
        scores = ops.matmul(q, k) * scale

        dtype = scores.dtype

        if self.local_attention_window is not None:
            mask = self._get_sliding_window_mask(t, dtype)
            scores = scores + (1.0 - mask[None, None]) * ops.cast(-1e9, dtype)

        if padding_mask is not None:
            pm = ops.cast(padding_mask, dtype)
            scores = scores + (1.0 - pm[:, None, None, :]) * ops.cast(
                -1e9, dtype
            )

        probs = ops.softmax(scores, axis=-1)
        probs = self.attn_dropout(probs, training=training)

        out = ops.matmul(probs, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, t, self.hidden_dim))

        return self.output_dense(out)

    def get_config(self):
        """Returns the serialization configuration of the attention layer."""
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "local_attention_window": self.local_attention_window,
                "dropout": self.dropout,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertEncoderLayer(layers.Layer):
    """ModernBERT encoder block.

    This layer encapsulates an instance of `ModernBertAttention`, followed by a
    `ModernBertMLP` block. Residual blocks are wrapped using Pre-Layer
    Normalization routing implemented through `RMSNormalization`.

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
        rotary_embedding=None,
        local_attention_window=None,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.local_attention_window = local_attention_window
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.attn_norm = layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
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

        self.mlp_norm = layers.RMSNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="mlp_norm",
        )

        self.mlp = ModernBertMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dtype=dtype,
            name="mlp",
        )

        self.attn_dropout = layers.Dropout(dropout, dtype=dtype)
        self.mlp_dropout = layers.Dropout(dropout, dtype=dtype)

    def call(self, x, padding_mask=None, training=None):
        """Forward pass of the complete encoder layer block."""
        residual = x

        x = self.attn_norm(x)
        x = self.attn(x, padding_mask=padding_mask, training=training)
        x = self.attn_dropout(x, training=training)

        if x.dtype != residual.dtype:
            x = ops.cast(x, residual.dtype)
        x = residual + x

        residual = x

        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.mlp_dropout(x, training=training)

        if x.dtype != residual.dtype:
            x = ops.cast(x, residual.dtype)
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
                "local_attention_window": self.local_attention_window,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
