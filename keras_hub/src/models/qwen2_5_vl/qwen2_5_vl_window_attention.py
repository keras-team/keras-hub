import keras
from keras import ops
from qwen2_5_vl_mrope import Qwen2_5_VLMRoPE3D
from qwen2_5_vl_global_attention import rotate_half


def partition_windows(x, window_size):
    shape = ops.shape(x)
    B = shape[0]
    T = shape[1]
    H = shape[2]
    W = shape[3]
    C = shape[4]

    H_blocks = ops.floor_divide(H, window_size)
    W_blocks = ops.floor_divide(W, window_size)

    x = ops.reshape(x, (B, T, H_blocks, window_size, W_blocks, window_size, C))
    x = ops.transpose(x, (0, 2, 4, 1, 3, 5, 6))
    x = ops.reshape(x, (B * H_blocks * W_blocks, T * window_size * window_size, C))
    return x


def reverse_windows(windows, B, T, H, W, C, window_size):
    H_blocks = ops.floor_divide(H, window_size)
    W_blocks = ops.floor_divide(W, window_size)

    x = ops.reshape(windows, (B, H_blocks, W_blocks, T, window_size, window_size, C))
    x = ops.transpose(x, (0, 3, 1, 4, 2, 5, 6))
    x = ops.reshape(x, (B, T, H, W, C))
    return x


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLWindowAttention(keras.layers.Layer):
    """
    Window-partitioned self-attention for Qwen2.5-VL vision blocks.

    Tokens are partitioned into non-overlapping spatial windows before
    attention is computed, reducing complexity from O(N²) to O(N·w²)
    where w is the window size.

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    window_size : int
    theta : float
        RoPE base frequency.
    """

    def __init__(self, hidden_size, num_heads, window_size=8, theta=10000.0, **kwargs):
        super().__init__(**kwargs)

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = hidden_size // num_heads
        self.theta    = theta

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
        C = self.hidden_size

        x = ops.reshape(x, (B, T, H, W, C))
        windows = partition_windows(x, self.window_size)

        win_shape = ops.shape(windows)
        BN = win_shape[0]
        Nw = win_shape[1]

        qkv = self.qkv(windows)
        qkv = ops.reshape(qkv, (BN, Nw, 3, self.num_heads, self.head_dim))

        q = ops.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        k = ops.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = ops.transpose(qkv[:, :, 2], (0, 2, 1, 3))

        cos, sin = self.mrope(T=T, H=self.window_size, W=self.window_size)
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
        out = ops.reshape(out, (BN, Nw, self.hidden_size))

        x = reverse_windows(out, B, T, H, W, C, self.window_size)
        x = ops.reshape(x, (B, T * H * W, self.hidden_size))
        return self.proj(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads":   self.num_heads,
            "window_size": self.window_size,
            "theta":       self.theta,
        })
        return config