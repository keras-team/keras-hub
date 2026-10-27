import keras
from keras import ops

from qwen2_5_vl_rms_norm import Qwen2_5_VLRMSNorm


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLSwiGLU(keras.layers.Layer):
    """
    SwiGLU feed-forward network for Qwen2.5-VL decoder blocks.

    Implements: down_proj(silu(gate_proj(x)) * up_proj(x))
    No bias on any projection, matching HF decoder MLP weights.

    Parameters
    ----------
    hidden_size : int
    intermediate_size : int
    """

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def build(self, input_shape):
        self.gate_proj = keras.layers.Dense(
            self.intermediate_size, use_bias=False, name="gate_proj"
        )
        self.up_proj = keras.layers.Dense(
            self.intermediate_size, use_bias=False, name="up_proj"
        )
        self.down_proj = keras.layers.Dense(
            self.hidden_size, use_bias=False, name="down_proj"
        )
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        self.down_proj.build(list(input_shape[:-1]) + [self.intermediate_size])
        super().build(input_shape)

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

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.kv_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

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

        q = ops.transpose(ops.reshape(q, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(k, (B, S, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(v, (B, S, self.num_kv_heads, self.head_dim)), (0, 2, 1, 3))

        k = ops.repeat(k, self.kv_groups, axis=1)
        v = ops.repeat(v, self.kv_groups, axis=1)

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        causal = ops.cast(ops.tril(ops.ones((S, S))), attn.dtype)
        attn = attn * causal + (1.0 - causal) * (-1e4)

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
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
        })
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLDecoderBlock(keras.layers.Layer):
    """
    Single Qwen2.5-VL decoder block with pre-norm architecture.

    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))

    Parameters
    ----------
    hidden_size : int
    num_heads : int
    num_kv_heads : int
    intermediate_size : int
    rms_epsilon : float
    dropout : float
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        intermediate_size,
        rms_epsilon=1e-6,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.rms_epsilon = rms_epsilon
        self.dropout = dropout

        self.input_norm = Qwen2_5_VLRMSNorm(
            hidden_size=hidden_size, epsilon=rms_epsilon, name="input_norm"
        )
        self.self_attn = Qwen2_5_VLGQAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            name="self_attn",
        )
        self.post_norm = Qwen2_5_VLRMSNorm(
            hidden_size=hidden_size, epsilon=rms_epsilon, name="post_norm"
        )
        self.mlp = Qwen2_5_VLSwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            name="mlp",
        )
        self.residual_dropout = (
            keras.layers.Dropout(dropout) if dropout > 0 else None
        )

    def call(self, x, attention_mask=None, training=False):
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(x, attention_mask=attention_mask, training=training)
        if self.residual_dropout is not None:
            x = self.residual_dropout(x, training=training)
        x = residual + x

        residual = x
        x = self.post_norm(x)
        x = self.mlp(x)
        if self.residual_dropout is not None:
            x = self.residual_dropout(x, training=training)
        x = residual + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "rms_epsilon": self.rms_epsilon,
            "dropout": self.dropout,
        })
        return config
