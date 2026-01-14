import keras
from keras import layers
from keras import ops
from keras_hub.src.utils.keras_utils import gelu_approximate

@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertMLP(layers.Layer):
    """ModernBERT MLP block using Gated Linear Units (GeGLU).

    This layer implements the gated linear unit MLP where the input is projected
    into two gated components. The output is calculated as:
    `output = wo(activation(wi_0(x)) * wi_1(x))`

    Args:
        hidden_dim: int. The input and output dimensionality of the layer.
        intermediate_dim: int. The dimensionality of the inner gated projections.
        activation: function. The activation function to use.
            Defaults to `keras_hub.src.utils.keras_utils.gelu_approximate`.
        **kwargs: Standard layer arguments.

    Examples:
    ```python
    import keras
    from keras import ops
    import keras_hub

    mlp = keras_hub.models.modernbert.ModernBertMLP(
        hidden_dim=768,
        intermediate_dim=1152,
    )
    x = ops.ones((2, 128, 768))
    output = mlp(x) # shape: (2, 128, 768)
    ```
    """

    def __init__(
        self, hidden_dim, intermediate_dim, activation=gelu_approximate, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = keras.activations.get(activation)

        self.wi_0 = layers.Dense(intermediate_dim, use_bias=False, name="wi_0")
        self.wi_1 = layers.Dense(intermediate_dim, use_bias=False, name="wi_1")
        self.wo = layers.Dense(hidden_dim, use_bias=False, name="wo")

    def call(self, x):
        return self.wo(self.activation(self.wi_0(x)) * self.wi_1(x))

    def get_config(self):
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
    """ModernBERT Attention with RoPE and Alternating Window support.

    This layer implements the multi-head self-attention mechanism used in
    ModernBERT, featuring Rotary Positional Embeddings (RoPE) and optional
    sliding window (local) attention.

    Args:
        hidden_dim: int. The dimensionality of the input and output.
        num_heads: int. Number of attention heads.
        rotary_embedding: `keras.layers.Layer`. A RotaryEmbedding layer instance.
        local_attention_window: int. The window size for local sliding attention.
            If `None`, global attention is performed. Defaults to `None`.
        dropout: float. Dropout probability for attention scores.
            Defaults to `0.0`.
        **kwargs: Standard layer arguments.

    Examples:
    ```python
    import keras
    from keras import ops
    import keras_hub

    attention = keras_hub.models.modernbert.ModernBertAttention(hidden_dim=768,
    num_heads=12,
    local_attention_window=128,
    )
    x = ops.ones((2, 128, 768))
    output = attention(x)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        rotary_embedding=None,
        local_attention_window=None,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.rotary_embedding = rotary_embedding
        self.local_attention_window = local_attention_window
        self.dropout = dropout

        self.qkv = layers.Dense(hidden_dim * 3, use_bias=False, name="Wqkv")
        self.out_dense = layers.Dense(hidden_dim, use_bias=False, name="Wo")
        self.dropout_layer = layers.Dropout(dropout)

    def _get_sliding_window_mask(self, seq_len):
        positions = ops.arange(seq_len)
        distance = ops.abs(positions[:, None] - positions[None, :])
        mask = ops.cast(distance <= (self.local_attention_window // 2), "float32")
        return mask

    def call(self, x, padding_mask=None, training=None):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        qkv = self.qkv(x)
        qkv = ops.reshape(
            qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim)
        )
        q, k, v = ops.unstack(qkv, axis=2)

        if self.rotary_embedding:
            q = self.rotary_embedding(q)
            k = self.rotary_embedding(k)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.head_dim, x.dtype))

        mask = None
        if self.local_attention_window is not None:
            mask = self._get_sliding_window_mask(seq_len)
            mask = mask[None, None, :, :]

        if padding_mask is not None:
            p_mask = ops.cast(padding_mask, "float32")
            p_mask = p_mask[:, None, None, :]
            mask = mask * p_mask if mask is not None else p_mask

        if mask is not None:
            scores = scores + (1.0 - mask) * -1e9

        attn = ops.softmax(scores, axis=-1)
        attn = self.dropout_layer(attn, training=training)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.hidden_dim))
        return self.out_dense(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "rotary_embedding": layers.serialize(self.rotary_embedding),
                "local_attention_window": self.local_attention_window,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config["rotary_embedding"] is not None:
            config["rotary_embedding"] = layers.deserialize(
                config["rotary_embedding"]
            )
        return cls(**config)


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertEncoderLayer(layers.Layer):
    """ModernBERT Encoder Layer.

    A modernized Transformer block featuring pre-normalization, RMSNorm scaling,
    GeGLU activations, and alternating attention.

    Args:
        hidden_dim: int. The dimensionality of the input and output.
        intermediate_dim: int. The dimensionality of the MLP's inner gated layers.
        num_heads: int. Number of attention heads.
        rotary_embedding: `keras.layers.Layer`. A RotaryEmbedding layer instance.
        local_attention_window: int. Window size for local attention.
            Defaults to `None`.
        dropout: float. Dropout rate for residual connections and attention.
            Defaults to `0.0`.
        layer_norm_epsilon: float. Epsilon for the RMSNorm layers.
            Defaults to `1e-5`.
        **kwargs: Standard layer arguments.

    Examples:
    ```python
    import keras
    from keras import ops
    import keras_hub

    layer = keras_hub.models.modernbert.ModernBertEncoderLayer(
        hidden_dim=768,
        intermediate_dim=1152,
        num_heads=12,
    )
    x = ops.ones((2, 128, 768))
    output = layer(x)
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.local_attention_window = local_attention_window
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.attn_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon, rms_scaling=True, name="attn_norm"
        )
        self.attn = ModernBertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            rotary_embedding=rotary_embedding,
            local_attention_window=local_attention_window,
            dropout=dropout,
            name="attn",
        )
        self.mlp_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon, rms_scaling=True, name="mlp_norm"
        )
        self.mlp = ModernBertMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            name="mlp",
        )
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, x, padding_mask=None, training=None):
        res = x
        x = self.attn_norm(x)
        x = self.attn(x, padding_mask=padding_mask, training=training)
        x = res + self.dropout_layer(x, training=training)

        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = res + self.dropout_layer(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "local_attention_window": self.local_attention_window,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "rotary_embedding": layers.serialize(self.attn.rotary_embedding),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config["rotary_embedding"] is not None:
            config["rotary_embedding"] = layers.deserialize(
                config["rotary_embedding"]
            )
        return cls(**config)