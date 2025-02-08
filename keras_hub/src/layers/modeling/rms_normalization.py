import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.RMSNormalization")
class RMSNormalization(keras.layers.Layer):
    """Root Mean Square (RMS) Normalization layer.

    This layer normalizes the input tensor based on its RMS value and applies
    a learned scaling factor.

    Args:
        input_dim: int. The dimensionality of the input tensor.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.scale = self.add_weight(
            name="scale", shape=(input_dim,), initializer="ones"
        )

    def call(self, x):
        """Applies RMS normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            The RMS-normalized tensor of the same shape (batch_size, input_dim),
            scaled by the learned `scale` parameter.
        """
        x = ops.cast(x, float)
        rrms = ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + 1e-6)
        return (x * rrms) * self.scale
