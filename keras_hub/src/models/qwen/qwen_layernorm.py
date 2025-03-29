import keras
from keras import ops


class QwenLayerNorm(keras.layers.Layer):
    """A normalization layer for Qwen that implements RMS normalization."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
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
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(x * self.scale, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
