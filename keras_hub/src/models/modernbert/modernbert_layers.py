import keras
from keras import layers
from keras import ops
from keras_hub.src.utils.keras_utils import gelu_approximate

@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertMLP(layers.Layer):
    """ModernBERT MLP block using Gated Linear Units (GeGLU).

    This block implements the gated activation mechanism: 
    `output = wo(activation(wi_0(x)) * wi_1(x))`.

    Args:
        hidden_dim: int. The input and output dimensionality of the block.
        intermediate_dim: int. The dimensionality of the inner gated projection.
        activation: function. The activation function to use (default: gelu).
        **kwargs: Standard layer arguments.
    """
    def __init__(self, hidden_dim, intermediate_dim, activation=gelu_approximate, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        
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
        })
        return config


@keras.utils.register_keras_serializable(package="keras_hub")
class ModernBertAttention(layers.Layer):
    """ModernBERT Attention with RoPE and Sliding Window support.

    This layer implements the multi-head attention mechanism used in ModernBERT,
    supporting both global attention and local sliding window attention.

    Args:
        hidden_dim: int. Dimensionality of the hidden state.
        num_heads: int. Number of attention heads.
        rotary_embedding: `RotaryEmbedding` layer. The RoPE layer to apply.
        local_attention_window: int. The size of the sliding window. If None,
            global attention is used.
        **kwargs: Standard layer arguments.
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
        
        self.qkv = layers.Dense(hidden_dim * 3, use_bias=False, name="qkv")
        self.out_dense = layers.Dense(hidden_dim, use_bias=False, name="out_dense")

    def _get_sliding_window_mask(self, seq_len):
        """Generates a boolean mask for sliding window attention."""
        # Create a distance matrix where dist[i, j] = |i - j|
        idx = ops.arange(seq_len)
        cols = idx[None, :]
        rows = idx[:, None]
        distance = ops.abs(rows - cols)
        
        mask = distance <= (self.local_attention_window // 2)
        return ops.cast(mask, dtype="float32")

    def call(self, x, padding_mask=None):
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        
        # Linear Projection
        qkv = self.qkv(x) 
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = ops.unstack(qkv, axis=2)

        # Rotary Position Embeddings (RoPE)
        if self.rotary_embedding is not None:
            q = self.rotary_embedding(q)
            k = self.rotary_embedding(k)

        # Transpose for attention calculation
        q = ops.transpose(q, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)
        k = ops.transpose(k, (0, 2, 3, 1))  # (batch, heads, head_dim, seq)
        v = ops.transpose(v, (0, 2, 1, 3))  # (batch, heads, seq, head_dim)

        scale = ops.cast(ops.sqrt(ops.cast(self.head_dim, x.dtype)), x.dtype)
        scores = ops.matmul(q, k) / scale
    
        # Local or Global Masks
        if self.local_attention_window is not None:
            sw_mask = self._get_sliding_window_mask(seq_len)
            # Add mask to scores (0 for attend, -inf for ignore)
            scores = scores + (1.0 - sw_mask[None, None, :, :]) * -1e9

        if padding_mask is not None:
            # Expand padding mask to (batch, 1, 1, seq)
            p_mask = ops.cast(padding_mask[:, None, None, :], scores.dtype)
            scores = scores + (1.0 - p_mask) * -1e9
        
        # softmax
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
    """ModernBERT Encoder Layer implementation.

    This layer implements a modernized Transformer block featuring:
    1. Pre-Normalization (Norm-First architecture).
    2. RMSNorm scaling (LayerNorm without additive bias).
    3. Gated Linear Unit (GeGLU) activation in the MLP.
    4. Alternating Attention (Global vs. Local Sliding Window).
    """
    def __init__(
        self, 
        hidden_dim, 
        intermediate_dim, 
        num_heads, 
        rotary_embedding=None,
        local_attention_window=None,
        dropout=0.0, 
        activation=gelu_approximate, 
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
            epsilon=layer_norm_epsilon, rms_scaling=True
        )
        self.attn = ModernBertAttention(
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            rotary_embedding=rotary_embedding,
            local_attention_window=local_attention_window
        )
        
        self.mlp_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon, rms_scaling=True
        )
        self.mlp = ModernBertMLP(
            hidden_dim=hidden_dim, 
            intermediate_dim=intermediate_dim, 
            activation=activation
        )
        self.dropout_layer = layers.Dropout(dropout)

    @property
    def is_global(self):
        # Returns True if the layer uses global attention.
        return self.local_attention_window is None
    
    def call(self, x, padding_mask=None):
        # Attention Residual path
        res = x
        x = self.attn_norm(x)
        x = self.attn(x, padding_mask=padding_mask)
        x = res + self.dropout_layer(x)
        
        # MLP Residual path
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