import keras
from keras import layers
from keras import ops

from keras_hub.src.models.flux.flux_maths import rearrange_symbolic_tensors
from keras_hub.src.models.flux.flux_maths import scaled_dot_product_attention


class MLP(keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        activation="gelu",
        dtype=None,
        **kwargs,
    ):
        super(MLP, self).__init__(**kwargs)
        self.Wi = layers.Dense(
            intermediate_size * 2,
            use_bias=False,
            dtype=dtype,
        )
        self.act = keras.activations.get(activation)
        self.Wo = layers.Dense(
            hidden_size,
            use_bias=False,
            dtype=dtype,
        )

    def call(self, x):
        input, gate = ops.split(self.Wi(x), 2, axis=-1)
        return self.Wo(self.act(input) * gate)


class ModernBERTAttention(keras.Model):
    def __init__(
        self, hidden_size, num_heads, rotary_embedding, dtype=None, **kwargs
    ):
        super(ModernBERTAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.rotary_embedding = rotary_embedding
        self.Wqkv = layers.Dense(hidden_size * 3, use_bias=False, dtype=dtype)
        self.Wo = layers.Dense(hidden_size, use_bias=False, dtype=dtype)

    def build(self, input_shape):
        self.Wqkv.build(input_shape)
        self.Wo.build((None, input_shape[1], input_shape[-1]))

    def call(self, x):
        qkv = self.Wqkv(x)
        q, k, v = rearrange_symbolic_tensors(qkv, K=3, H=self.num_heads)

        # Apply rotary embeddings
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        # Apply scaled dot product attention
        x = scaled_dot_product_attention(q, k, v)

        # Reshape and apply final dense layer
        x = ops.transpose(x, (0, 2, 1, 3))
        b, s, h, d = ops.shape(x)
        x = ops.reshape(x, (b, s, h * d))
        x = self.Wo(x)
        return x


class ModernBERTEncoderLayer(keras.Model):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_heads,
        activation="gelu",
        layer_norm_epsilon=1e-05,
        rotary_embedding=None,
        dtype=None,
        **kwargs,
    ):
        super(ModernBERTEncoderLayer, self).__init__(**kwargs)
        self.attn = ModernBERTAttention(
            hidden_size, num_heads, rotary_embedding, dtype=dtype
        )
        self.mlp_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon, dtype=dtype
        )
        self.mlp = MLP(hidden_size, intermediate_size, activation, dtype=dtype)

    def call(self, x):
        x = self.attn(x)
        x = self.mlp_norm(x)
        x = self.mlp(x)
        return x
