import keras
from keras import ops
from qwen2_5_vl_mrope import Qwen2_5_VLMRoPE3D


def rotate_half(x):
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    return ops.concatenate((-x2, x1), axis=-1)


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLGlobalAttention(keras.layers.Layer):
    """
    Full (global) self-attention for Qwen2.5-VL vision blocks.

    Used at specific block indices (fullatt_block_indexes) where
    all tokens attend to all other tokens without window partitioning.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    theta : float
        RoPE base frequency.
    """

    def __init__(self, hidden_size, num_heads, theta=10000.0, **kwargs):
        super().__init__(**kwargs)

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.theta = theta

        self.mrope = Qwen2_5_VLMRoPE3D(
            head_dim=self.head_dim,
            theta=theta,
            mrope_section=[16, 24, 24] if self.head_dim >= 64 else None,
        )
        self.qkv = keras.layers.Dense(hidden_size * 3, use_bias=True, name="qkv")
        self.proj = keras.layers.Dense(hidden_size, use_bias=True, name="proj")

    def call(self, x, *, T, H, W, training=False):
        shape = ops.shape(x)
        B = shape[0]
        N = shape[1]

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))

        q = ops.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        k = ops.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = ops.transpose(qkv[:, :, 2], (0, 2, 1, 3))

        cos, sin = self.mrope(T=T, H=H, W=W)
        cos = ops.expand_dims(ops.expand_dims(cos, axis=0), axis=0)
        sin = ops.expand_dims(ops.expand_dims(sin, axis=0), axis=0)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        scale = ops.rsqrt(ops.cast(self.head_dim, x.dtype))
        q = q * scale

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        attn = ops.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (B, N, self.hidden_size))

        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "theta": self.theta,
        })
        return config