import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.models.Qwen3OmniMoeLayerNorm")
class Qwen3OmniMoeLayerNorm(keras.layers.Layer):
    """Layer normalization for Qwen3-Omni MoE model."""

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
        self.beta = self.add_weight(
            name="beta",
            shape=(input_shape[-1],),
            initializer="zeros",
            dtype=self.dtype,
        )
        super().build(input_shape)

    def call(self, inputs):
        # Compute mean and variance
        mean = ops.mean(inputs, axis=-1, keepdims=True)
        variance = ops.var(inputs, axis=-1, keepdims=True)

        # Normalize
        normalized = (inputs - mean) / ops.sqrt(variance + self.epsilon)

        # Scale and shift
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
            }
        )
        return config
