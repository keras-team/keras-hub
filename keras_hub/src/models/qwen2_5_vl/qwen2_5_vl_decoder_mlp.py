import keras
from keras import ops


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
        self.hidden_size       = hidden_size
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
            "hidden_size":       self.hidden_size,
            "intermediate_size": self.intermediate_size,
        })
        return config