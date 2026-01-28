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
        head_dim: int. The dimension of each attention head, used for per-head
            normalization. Defaults to `None`.
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
    
    # For per-head normalization in attention
    layer = Qwen3OmniMoeLayerNorm(head_dim=64, epsilon=1e-6)
    ```
    """

    def __init__(
        self,
        head_dim=None,
        epsilon=1e-6,
        **kwargs,
    ):
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

    def call(self, inputs):
        input_dtype = inputs.dtype
        x = ops.cast(inputs, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(x * self.scale, input_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "epsilon": self.epsilon,
            }
        )
        return config
