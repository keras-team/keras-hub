import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLRMSNorm(keras.layers.Layer):
    """
    Root Mean Square Layer Normalization for Qwen2.5-VL.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the weight vector.
    epsilon : float
        Small constant for numerical stability.
    """

    def __init__(self, hidden_size, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(self.hidden_size,),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        variance = ops.mean(ops.square(inputs), axis=-1, keepdims=True)
        normalized = inputs * ops.rsqrt(variance + self.epsilon)
        return normalized * self.weight

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "epsilon": self.epsilon,
        })
        return config
