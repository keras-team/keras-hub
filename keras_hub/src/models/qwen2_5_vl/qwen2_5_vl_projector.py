import keras
from keras import ops


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLVisionProjector(keras.layers.Layer):
    """
    Projects vision encoder outputs into the LLM hidden space.

    Two-layer MLP with GELU activation:
        x → Dense(intermediate_dim) → GELU → Dense(text_hidden_size)

    Parameters
    ----------
    vision_hidden_size : int
        Input dimensionality (output of PatchMerger = 4 * vision_hidden_size).
    text_hidden_size : int
        Output dimensionality matching the LLM hidden size.
    intermediate_multiplier : int
        Multiplier for the intermediate layer size.
    """

    def __init__(
        self,
        vision_hidden_size,
        text_hidden_size,
        intermediate_multiplier=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.intermediate_multiplier = intermediate_multiplier
        self.intermediate_dim = vision_hidden_size * intermediate_multiplier

        self.dense1 = keras.layers.Dense(
            self.intermediate_dim, use_bias=True, name="dense1"
        )
        self.dense2 = keras.layers.Dense(
            text_hidden_size, use_bias=True, name="dense2"
        )

    def build(self, input_shape):
        self.dense1.build(input_shape)
        self.dense2.build((input_shape[0], input_shape[1], self.intermediate_dim))
        super().build(input_shape)

    def call(self, x):
        x = self.dense1(x)
        x = ops.gelu(x)
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.text_hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vision_hidden_size": self.vision_hidden_size,
            "text_hidden_size": self.text_hidden_size,
            "intermediate_multiplier": self.intermediate_multiplier,
        })
        return config
