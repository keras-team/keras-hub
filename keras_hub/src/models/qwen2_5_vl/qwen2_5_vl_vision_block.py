import keras
from keras import ops

from qwen2_5_vl_window_attention import Qwen2_5_VLWindowAttention
from qwen2_5_vl_global_attention import Qwen2_5_VLGlobalAttention
from qwen2_5_vl_rms_norm import Qwen2_5_VLRMSNorm


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLVisionSwiGLU(keras.layers.Layer):
    """
    SwiGLU MLP for Qwen2.5-VL vision blocks.

    All projections include bias, matching HF visual.blocks.{i}.mlp weights.

    Parameters
    ----------
    hidden_size : int
    intermediate_size : int
    """

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = keras.layers.Dense(
            intermediate_size, use_bias=True, name="gate_proj"
        )
        self.up_proj = keras.layers.Dense(
            intermediate_size, use_bias=True, name="up_proj"
        )
        self.down_proj = keras.layers.Dense(
            hidden_size, use_bias=True, name="down_proj"
        )

    def call(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        })
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLVisionBlock(keras.layers.Layer):
    """
    Hybrid vision transformer block for Qwen2.5-VL.

    Uses window attention by default, or global attention when
    use_global_attention=True (at fullatt_block_indexes).

    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    intermediate_size : int
    window_size : int
    use_global_attention : bool
    theta : float
    rms_eps : float
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        window_size,
        use_global_attention=False,
        theta=10000.0,
        rms_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.use_global_attention = use_global_attention
        self.theta = theta
        self.rms_eps = rms_eps

        self.norm1 = Qwen2_5_VLRMSNorm(hidden_size, epsilon=rms_eps, name="norm1")
        self.norm2 = Qwen2_5_VLRMSNorm(hidden_size, epsilon=rms_eps, name="norm2")

        if use_global_attention:
            self.attn = Qwen2_5_VLGlobalAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                theta=theta,
                name="attn",
            )
        else:
            self.attn = Qwen2_5_VLWindowAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                window_size=window_size,
                theta=theta,
                name="attn",
            )

        self.mlp = Qwen2_5_VLVisionSwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            name="mlp",
        )

    def call(self, x, *, T, H, W, training=False):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, T=T, H=H, W=W, training=training)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "window_size": self.window_size,
            "use_global_attention": self.use_global_attention,
            "theta": self.theta,
            "rms_eps": self.rms_eps,
        })
        return config
