import keras
from keras import ops


class Qwen3_5LayerNorm(keras.layers.Layer):
    """RMS normalization layer for Qwen3.5.

    Qwen3.5 uses a (1 + weight)-centered RMSNorm. Weights are initialized
    to zero so the effective scale starts at 1.0.
    """

    def __init__(self, head_dim=None, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = self.head_dim if self.head_dim else input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="zeros",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        input_dtype = x.dtype
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(x * (1.0 + self.scale), input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "epsilon": self.epsilon,
            }
        )
        return config
