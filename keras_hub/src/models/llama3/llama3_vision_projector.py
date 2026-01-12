import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.models.Llama3VisionProjector")
class Llama3VisionProjector(keras.layers.Layer):
    """Vision projector for the Llama 3.2 Vision model.

    This layer projects vision encoder features into the text embedding space
    using a two-layer MLP, enabling vision-language fusion.

    Args:
        hidden_dim: int. The dimension of the vision encoder output.
        output_dim: int. The dimension of the text decoder embeddings.
        intermediate_dim: int. The intermediate MLP dimension.
            Defaults to `output_dim`.
        activation: str. The activation function. Defaults to `"gelu"`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        hidden_dim,
        output_dim,
        intermediate_dim=None,
        activation="gelu",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # === Config ===
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim or output_dim
        self.activation = activation

        # === Layers ===
        self.dense_1 = layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            name="dense_1",
        )
        self.dense_2 = layers.Dense(
            self.output_dim,
            name="dense_2",
        )

    def build(self, input_shape):
        self.dense_1.build(input_shape)
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self.dense_2.build(tuple(intermediate_shape))
        super().build(input_shape)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
            }
        )
        return config
