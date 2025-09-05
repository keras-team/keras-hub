import keras
from keras import ops


class GptOssLayerNormalization(keras.layers.Layer):
    """A normalization layer for GPT-OSS that implements RMS normalization.

    This layer applies Root Mean Square (RMS) normalization, which is a common
    normalization technique used in models like Llama and GPT-OSS. It normalizes
    the input by its root mean square, then scales it by a learnable weight.

    Args:
        epsilon: A small float number to prevent division by zero.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # The last dimension of the input is the feature dimension.
        dim = input_shape[-1]
        # Create a learnable scale parameter, initialized to ones.
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        # Cast the input to float32 for numerical stability during computation,
        # similar to the PyTorch implementation's `hidden_states.to(torch.float32)`.
        x = ops.cast(x, "float32")

        # Calculate the variance (mean of squared values) along the last axis.
        # `keepdims=True` ensures the output shape is compatible for broadcasting.
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)

        # Apply RMS normalization: x / sqrt(variance + epsilon)
        x = x * ops.rsqrt(var + self.epsilon)

        # Scale the normalized input by the learnable `self.scale` parameter
        # and cast it back to the layer's compute dtype.
        # This matches the PyTorch implementation's `(self.weight * hidden_states).to(input_dtype)`.
        return ops.cast(x * self.scale, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


__all__ = ["GptOssLayerNormalization"]