import keras
from keras import layers

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.models.Llama3VisionProjector")
class Llama3VisionProjector(keras.layers.Layer):
    """The Vision Projector for the Llama 3 Vision model.

    This layer projects the output of the Vision Encoder (visual features)
    into the embedding space of the Text Decoder. It acts as the "bridge"
    between the vision and language modalities.

    Args:
        hidden_dim: int. The output dimension of the vision encoder.
        output_dim: int. The dimension of the text decoder embeddings
            (e.g., 4096).
        intermediate_dim: int. The size of the hidden layer in the
            projection MLP. If None, defaults to `output_dim`.
        activation: string or `keras.activations`. The activation function
            to use in the MLP. Defaults to "gelu".
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for layer computations and weights.
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
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim or output_dim
        self.activation = activation

        # The Projector is usually a 2-layer MLP:
        # 1. Project Up/Process
        self.dense_1 = layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            name="dense_1",
        )
        # 2. Project to Text Dimension
        self.dense_2 = layers.Dense(
            self.output_dim,
            name="dense_2",
        )

    def build(self, input_shape):
        # input_shape will be (batch, num_patches, hidden_dim)
        self.dense_1.build(input_shape)

        # dense_1 output shape: (batch, num_patches, intermediate_dim)
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
