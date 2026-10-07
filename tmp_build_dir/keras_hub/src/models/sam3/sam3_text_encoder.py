import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.models.sam3.sam3_utils import create_bidirectional_mask


class CLIPEncoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        intermediate_activation="gelu",
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.intermediate_activation = intermediate_activation
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.layer_norm_1 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm_1",
        )
        self.attention = layers.MultiHeadAttention(
            num_heads,
            hidden_dim // num_heads,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm_2 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm_2",
        )
        self.dense_1 = layers.Dense(
            self.intermediate_dim, dtype=self.dtype_policy, name="dense_1"
        )
        self.activation = layers.Activation(
            intermediate_activation, dtype=self.dtype_policy, name="activation"
        )
        self.dense_2 = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="dense_2"
        )

    def build(self, inputs_shape, attention_mask_shape):
        self.layer_norm_1.build(inputs_shape)
        self.attention.build(inputs_shape, inputs_shape, inputs_shape)
        self.layer_norm_2.build(inputs_shape)
        self.dense_1.build(inputs_shape)
        input_shape = self.dense_1.compute_output_shape(inputs_shape)
        self.dense_2.build(input_shape)

    def compute_output_shape(self, inputs_shape, attention_mask_shape):
        outputs_shape = list(inputs_shape)
        outputs_shape[-1] = self.hidden_dim
        return outputs_shape

    def call(self, inputs, attention_mask, training=None):
        residual = inputs
        x = self.layer_norm_1(inputs)
        x = self.attention(
            x,
            x,
            x,
            attention_mask=ops.cast(attention_mask, dtype="bool"),
            training=training,
        )
        x = ops.add(residual, x)

        residual = x
        x = self.dense_1(self.layer_norm_2(residual))
        x = self.activation(x)
        x = self.dense_2(x)
        x = ops.add(residual, x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


@keras_hub_export("keras_hub.layers.SAM3TextEncoder")
class SAM3TextEncoder(layers.Layer):
    """A text encoder for the Segment Anything Model 3 (SAM3).

    This layer implements a CLIP-style text encoder. It processes token IDs and
    padding masks to produce text embeddings that are used as prompts for
    segmentation.

    Args:
        vocabulary_size: int. The size of the vocabulary.
        embedding_dim: int. The dimension of the token embeddings.
        hidden_dim: int. The hidden dimension of the transformer layers.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The dimension of the intermediate layer in the
            transformer's MLP.
        intermediate_activation: str. The activation function for the
            transformer layers. Defaults to `"gelu"`.
        max_sequence_length: int. The maximum sequence length. Defaults to
            `32`.
        layer_norm_epsilon: float. The epsilon value for layer normalization.
            Defaults to `1e-6`.
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        intermediate_activation="gelu",
        max_sequence_length=32,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocabulary_size = int(vocabulary_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.intermediate_activation = intermediate_activation
        self.max_sequence_length = int(max_sequence_length)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.embedding = TokenAndPositionEmbedding(
            vocabulary_size=self.vocabulary_size,
            sequence_length=self.max_sequence_length,
            embedding_dim=self.embedding_dim,
            dtype=self.dtype_policy,
            name="embedding",
        )
        self.encoder_layers = [
            CLIPEncoderLayer(
                self.hidden_dim,
                self.num_heads,
                self.intermediate_dim,
                self.intermediate_activation,
                dtype=self.dtype_policy,
                name=f"encoder_layer_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.layer_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )

    def build(self, token_ids_shape, padding_masks_shape):
        self.embedding.build(token_ids_shape)
        x_shape = self.embedding.compute_output_shape(token_ids_shape)
        for layer in self.encoder_layers:
            layer.build(x_shape, padding_masks_shape)
        self.layer_norm.build(x_shape)

    def call(self, token_ids, padding_masks, training=None):
        x = self.embedding(token_ids, training=training)
        padding_masks = create_bidirectional_mask(x, padding_masks)
        for layer in self.encoder_layers:
            x = layer(x, padding_masks, training=training)
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "intermediate_activation": self.intermediate_activation,
                "max_sequence_length": self.max_sequence_length,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(self, token_ids_shape, padding_masks_shape):
        return self.embedding.compute_output_shape(token_ids_shape)

    def compute_output_spec(self, token_ids, padding_masks):
        output_shape = self.compute_output_shape(
            token_ids.shape, padding_masks.shape
        )
        return keras.KerasTensor(output_shape, dtype=self.compute_dtype)
