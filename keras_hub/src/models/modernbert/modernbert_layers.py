import keras
from keras import layers
from keras import ops
from keras_hub.src.utils.keras_utils import gelu_approximate

@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertMLP(layers.Layer):
    """ModernBERT MLP block using Gated Linear Units (GeGLU).
    
    Implements: output = wo(activation(wi_0(x)) * wi_1(x)).
    
    Args:
        hidden_dim: int. Input or output dimensionality.
        intermediate_dim: int. Inner gated projection dimensionality.
        activation: function. Activation function (default: gelu_approximate).
    """
    def __init__(self, hidden_dim, intermediate_dim, activation=gelu_approximate, **kwargs):
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
        config.update({
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "activation": keras.activations.serialize(self.activation),
        })
        return config


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertAttention(layers.Layer):
    """ModernBERT Attention with RoPE and Alternating Window support.
    
    Supports both global and local sliding window attention.
    """
    def __init__(
        self, 
        hidden_dim, 
        num_heads, 
        rotary_embedding=None, 
        local_attention_window=None, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.rotary_embedding = rotary_embedding
        self.local_attention_window = local_attention_window
        
        self.qkv = layers.Dense(hidden_dim * 3, use_bias=False, name="Wqkv")
        self.out_dense = layers.Dense(hidden_dim, use_bias=False, name="Wo")

    def _get_sliding_window_mask(self, seq_len):
        idx = ops.arange(seq_len)
        distance = ops.abs(idx[:, None] - idx[None, :])
        mask = distance <= (self.local_attention_window // 2)
        return ops.cast(mask, dtype="float32")

    def call(self, x, padding_mask=None):
        batch_size, seq_len = ops.shape(x)[0], ops.shape(x)[1]
        
        qkv = self.qkv(x) 
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = ops.unstack(qkv, axis=2)

        if self.rotary_embedding:
            q, k = self.rotary_embedding(q), self.rotary_embedding(k)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 3, 1))
        v = ops.transpose(v, (0, 2, 1, 3))

        scores = ops.matmul(q, k) / ops.sqrt(ops.cast(self.head_dim, x.dtype))

        # ==== Sliding Window Mask ====
        if self.local_attention_window is not None:
            sw_mask = self._get_sliding_window_mask(seq_len)
            scores += (1.0 - sw_mask[None, None, :, :]) * -1e9
        
        if padding_mask is not None:
            p_mask = ops.cast(padding_mask, x.dtype)
            p_mask = p_mask[:, None, None, :]
            scores += (1.0 - p_mask) * -1e9

        attn = ops.softmax(scores, axis=-1)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.hidden_dim))
        return self.out_dense(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "local_attention_window": self.local_attention_window,
        })
        return config


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertEncoderLayer(layers.Layer):
    """
    ModernBERT Encoder Layer.
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
        **kwargs
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
            hidden_dim, num_heads, rotary_embedding, local_attention_window, name="attn"
        )
        self.mlp_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon, rms_scaling=True, name="mlp_norm"
        )
        self.mlp = ModernBertMLP(hidden_dim, intermediate_dim, name="mlp")
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, x, padding_mask=None):
        res = x
        x = self.attn_norm(x)
        x = self.attn(x, padding_mask=padding_mask)
        x = res + self.dropout_layer(x)
        
        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = res + self.dropout_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "num_heads": self.num_heads,
            "local_attention_window": self.local_attention_window,
            "dropout": self.dropout,
            "layer_norm_epsilon": self.layer_norm_epsilon,
        })
        return config
