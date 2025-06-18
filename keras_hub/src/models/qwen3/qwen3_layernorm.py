import keras
from keras import ops


class Qwen3LayerNorm(keras.layers.Layer):
    """A normalization layer for Qwen that implements RMS normalization."""

    def __init__(self, head_dim=None, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.epsilon = epsilon

    def build(self, input_shape):
        if self.head_dim:
            dim = self.head_dim
        else:
            dim = input_shape[-1]

        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(x * self.scale, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
