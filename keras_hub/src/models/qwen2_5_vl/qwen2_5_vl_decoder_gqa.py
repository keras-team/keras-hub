import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLGQAAttention(keras.layers.Layer):
    """
    Grouped Query Attention for Qwen2.5-VL decoder blocks.

    Q/K/V projections have bias; O projection has no bias,
    matching the HF checkpoint exactly.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
        Number of query heads.
    num_kv_heads : int
        Number of key/value heads. Must divide num_heads evenly.
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, **kwargs):
        super().__init__(**kwargs)

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        self.hidden_size  = hidden_size
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = hidden_size // num_heads
        self.kv_groups    = num_heads // num_kv_heads
        self.scale        = self.head_dim ** -0.5

    def build(self, input_shape):
        self.q_proj = keras.layers.Dense(
            self.hidden_size, use_bias=True, name="q_proj"
        )
        self.k_proj = keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=True, name="k_proj"
        )
        self.v_proj = keras.layers.Dense(
            self.num_kv_heads * self.head_dim, use_bias=True, name="v_proj"
        )
        self.o_proj = keras.layers.Dense(
            self.hidden_size, use_bias=False, name="o_proj"
        )
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build(list(input_shape[:-1]) + [self.hidden_size])
        super().build(input_shape)

    def call(self, x, attention_mask=None, training=False):
        B = ops.shape(x)[0]
        S = ops.shape(x)[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = ops.transpose(
            ops.reshape(q, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        k = ops.transpose(
            ops.reshape(k, (B, S, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3)
        )
        v = ops.transpose(
            ops.reshape(v, (B, S, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3)
        )

        k = ops.repeat(k, self.kv_groups, axis=1)
        v = ops.repeat(v, self.kv_groups, axis=1)

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        causal = ops.cast(ops.tril(ops.ones((S, S))), attn.dtype)
        attn   = attn * causal + (1.0 - causal) * (-1e4)

        if attention_mask is not None:
            attn = attn + ops.cast(attention_mask, attn.dtype)

        attn = ops.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B, S, self.hidden_size))
        return self.o_proj(out)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size":  self.hidden_size,
            "num_heads":    self.num_heads,
            "num_kv_heads": self.num_kv_heads,
        })
        return config