import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.models.Qwen3OmniMoeLayerNorm")
class Qwen3OmniMoeLayerNorm(keras.layers.Layer):
    """RMS Normalization layer for Qwen3-Omni MoE model.

    RMSNorm (Root Mean Square Normalization) is a normalization technique
    that normalizes inputs by the root mean square of the inputs, without
    centering them around zero. This is commonly used in modern transformer
    architectures like Qwen models.

    Args:
        epsilon: float, default 1e-6. A small value added to the denominator
            for numerical stability.
        dtype: str or `keras.mixed_precision.DTypePolicy`. The dtype to use for
            the layer's computations and weights.

    Example:
    ```python
    # Create an RMSNorm layer
    layer = Qwen3OmniMoeLayerNorm(epsilon=1e-6)
    
    # Apply to input tensor
    inputs = keras.random.normal((2, 10, 128))
    outputs = layer(inputs)  # Shape: (2, 10, 128)
    ```
    """

    def __init__(
        self,
        epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(input_shape[-1],),
            initializer="ones",
            dtype=self.dtype,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute mean of squares
        variance = ops.mean(ops.square(inputs), axis=-1, keepdims=True)

        # Normalize
        normalized = inputs / ops.sqrt(variance + self.epsilon)

        # Scale
        return self.gamma * normalized

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
            }
        )
        return config