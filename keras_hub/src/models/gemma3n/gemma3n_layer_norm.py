import keras
from keras import ops


class Gemma3nRMSNorm(keras.layers.Layer):
    """RMS Normalization layer for Gemma3n."""

    def __init__(self, epsilon=1e-6, with_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.with_scale = with_scale

    def build(self, input_shape):
        if self.with_scale:
            self.scale = self.add_weight(
                name="scale",
                shape=(input_shape[-1],),
                initializer="ones",
                trainable=True,
            )
        else:
            self.scale = 1.0
        self.built = True

    def call(self, x):
        x_dtype = x.dtype
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")

        variance = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normalized_x = x * ops.rsqrt(variance + self.epsilon)
        output = normalized_x * scale
        return ops.cast(output, x_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon, "with_scale": self.with_scale})
        return config
