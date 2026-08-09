import keras
from keras import ops


class Gemma3nRMSNorm(keras.layers.Layer):
    """The Gemma 3n specific RMS normalization layer.

    Args:
        dim: int. The dimension of the input tensor.
        eps: float. A small constant added to the denominator for numerical
            stability. Defaults to `1e-6`.
        with_scale: bool. Whether to include a learnable scaling parameter.
            Defaults to `True`.
    """

    def __init__(self, dim, eps=1e-6, with_scale=True, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.eps = eps
        self.with_scale = with_scale

    def build(self, input_shape):
        if self.with_scale:
            self.scale = self.add_weight(
                shape=(self.dim,),
                initializer="ones",
                trainable=True,
                name="scale",
                dtype=self.dtype_policy.variable_dtype,
            )
        else:
            self.scale = 1.0
        super().build(input_shape)

    def call(self, x):
        norm_x = x * ops.rsqrt(
            ops.mean(ops.square(x), axis=-1, keepdims=True) + self.eps
        )
        return norm_x * self.scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "eps": self.eps,
                "with_scale": self.with_scale,
            }
        )
        return config
