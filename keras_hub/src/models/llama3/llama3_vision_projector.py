import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.models.Llama3VisionProjector")
class Llama3VisionProjector(keras.layers.Layer):
    """Vision projector for the Llama 3.2 Vision model.

    This layer projects vision encoder features into the text embedding space
    using a single linear projection, matching the HuggingFace architecture.

    Args:
        input_dim: int. The dimension of the vision encoder output
            (vision_output_dim from HuggingFace config, typically 7680).
        output_dim: int. The dimension of the text decoder embeddings.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # === Config ===
        self.input_dim = input_dim
        self.output_dim = output_dim

        # === Layers ===
        # Single linear projection matching HuggingFace nn.Linear
        self.projection = layers.Dense(
            self.output_dim,
            use_bias=True,
            name="projection",
        )

    def build(self, input_shape):
        self.projection.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.projection(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            }
        )
        return config
