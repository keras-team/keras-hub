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

        self.wi_0 = layers.Dense(intermediate_dim, use_bias=False, dtype="float16", name="wi_0")
        self.wi_1 = layers.Dense(intermediate_dim, use_bias=False, dtype="float16", name="wi_1") 
        self.wo = layers.Dense(hidden_dim, use_bias=False, dtype="float16", name="wo")
    
    def call(self, x):
        compute_dtype = self.compute_dtype
        # Gated Linear Unit math - ensure consistent dtypes
        gate = ops.cast(self.wi_0(x), compute_dtype)
        gate = self.activation(gate)
        value = ops.cast(self.wi_1(x), compute_dtype)
        return self.wo(gate * value)
    
    def compute_output_spec(self, x_spec, **kwargs):
        # MLP: hidden_dim -> intermediate_dim -> hidden_dim
        return keras.KerasTensor(
            shape=x_spec.shape, 
            dtype=self.compute_dtype
        )


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
    
    def __init__(self, hidden_dim, num_heads, rotary_embedding=None, local_attention_window=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.rotary_embedding = rotary_embedding
        self.local_attention_window = local_attention_window

        self.qkv = layers.Dense(hidden_dim * 3, use_bias=False, dtype="float16", name="Wqkv")
        self.out_dense = layers.Dense(hidden_dim, use_bias=False, dtype="float16", name="Wo")

    def _get_sliding_window_mask(self, seq_len, dtype):
        idx = ops.arange(seq_len)
        distance = ops.abs(idx[:, None] - idx[None, :])
        mask = distance <= (self.local_attention_window // 2)
        return ops.cast(mask, dtype=dtype)

    def call(self, x, padding_mask=None):
        batch_size, seq_len = ops.shape(x)[0], ops.shape(x)[1]
        compute_dtype = self.compute_dtype
        
        qkv = self.qkv(x) 
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = ops.unstack(qkv, axis=2)

        if self.rotary_embedding:
            q = ops.cast(self.rotary_embedding(q), compute_dtype)
            k = ops.cast(self.rotary_embedding(k), compute_dtype)

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 3, 1))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaling logic
        scale = ops.cast(ops.sqrt(ops.cast(self.head_dim, compute_dtype)), compute_dtype)
        scores = ops.matmul(q, k) / scale

        mask_value = ops.cast(-1e4, compute_dtype)
        one = ops.cast(1.0, compute_dtype)

        if self.local_attention_window is not None:
            sw_mask = self._get_sliding_window_mask(seq_len, compute_dtype)
            scores += (one - sw_mask[None, None, :, :]) * mask_value
        
        if padding_mask is not None:
            p_mask = ops.cast(padding_mask, compute_dtype)
            p_mask = p_mask[:, None, None, :]
            scores += (one - p_mask) * mask_value

        attn = ops.softmax(scores, axis=-1)
        attn = ops.cast(attn, compute_dtype)
        
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.hidden_dim))
        return self.out_dense(out)
    
    def compute_output_spec(self, x_spec, padding_mask_spec=None, **kwargs):
        # Attention preserves input shape and uses compute_dtype
        return keras.KerasTensor(shape=x_spec.shape, dtype=self.compute_dtype)


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
    def __init__(self, hidden_dim, intermediate_dim, num_heads, 
                 rotary_embedding=None, local_attention_window=None, 
                 dropout=0.0, layer_norm_epsilon=1e-5, **kwargs):
        
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.local_attention_window = local_attention_window
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.attn_norm = layers.LayerNormalization(epsilon=layer_norm_epsilon,dtype="float16", name="attn_norm")
        self.attn = ModernBertAttention(hidden_dim, num_heads, rotary_embedding,local_attention_window, name="attn")
        self.mlp_norm = layers.LayerNormalization(epsilon=layer_norm_epsilon,dtype="float16", name="mlp_norm")
        self.mlp = ModernBertMLP(hidden_dim,intermediate_dim, name="mlp")


        self.dropout_layer = layers.Dropout(dropout, dtype="float16")

    
    def call(self, x, padding_mask=None):
        compute_dtype = self.compute_dtype
    
        # Self-Attention block
        attn_res = self.attn_norm(x)      
        attn_out = self.attn(attn_res, padding_mask=padding_mask)
        x = attn_res + self.dropout_layer(attn_out)
    
        # MLP block  
        mlp_res = self.mlp_norm(x)        
        mlp_out = self.mlp(mlp_res)
        x = mlp_res + self.dropout_layer(mlp_out)  
    
        return x

    
    def compute_output_spec(self, x_spec, **kwargs):
        padding_mask_spec = kwargs.get('padding_mask_spec', None)
    
        # Attention path
        attn_res_spec = self.attn_norm.compute_output_spec(x_spec)
        attn_spec = self.attn.compute_output_spec(attn_res_spec, padding_mask_spec=padding_mask_spec)
        x_spec = self.dropout_layer.compute_output_spec(attn_spec)
    
        # MLP path
        mlp_res_spec = self.mlp_norm.compute_output_spec(x_spec)
        mlp_spec = self.mlp.compute_output_spec(mlp_res_spec)
        return self.dropout_layer.compute_output_spec(mlp_spec)
        
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