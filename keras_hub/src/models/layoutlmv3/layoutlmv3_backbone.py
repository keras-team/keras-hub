"""LayoutLMv3 backbone implementation."""

import keras
from keras import ops
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.models.backbone import Backbone


@keras.saving.register_keras_serializable(package="keras_hub")
class LayoutLMv3Backbone(Backbone):
    """LayoutLMv3 backbone model.

    This model implements the LayoutLMv3 architecture for document understanding
    tasks. It processes text tokens along with their spatial (bounding box)
    information to generate contextualized representations.

    Args:
        vocabulary_size: int. The size of the vocabulary.
        hidden_dim: int. The size of the hidden dimension.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The size of the intermediate dimension in the
            feed-forward network.
        dropout: float. The dropout rate.
        max_sequence_length: int. The maximum sequence length.
        type_vocab_size: int. The size of the token type vocabulary.
        initializer_range: float. The standard deviation of the initializer.
        layer_norm_epsilon: float. The epsilon value for layer normalization.
        patch_size: int. The size of the image patches.
        image_size: int. The size of the input images.
        num_channels: int. The number of input image channels.
        dtype: str. The dtype of the model.

    Example:

    ```python
    # Create a LayoutLMv3 backbone
    model = LayoutLMv3Backbone(
        vocabulary_size=30522,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
    )

    # Call the model
    input_data = {
        "token_ids": tf.constant([[1, 2, 3, 4, 5]]),
        "padding_mask": tf.constant([[1, 1, 1, 1, 1]]),
        "bbox": tf.constant([[[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10], [30, 0, 40, 10], [40, 0, 50, 10]]]),
    }
    output = model(input_data)
    ```

    References:
        - [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
        - [LayoutLMv3 GitHub](https://github.com/microsoft/unilm/tree/master/layoutlmv3)
    """

    def __init__(
        self,
        vocabulary_size=30522,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_epsilon=1e-12,
        patch_size=16,
        image_size=224,
        num_channels=3,
        spatial_embedding_dim=None,
        dtype="float32",
        **kwargs,
    ):
        # Store configuration
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.spatial_embedding_dim = spatial_embedding_dim or hidden_dim // 4

        # Token embedding layer
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            dtype=dtype,
            name="token_embedding",
        )

        # Position embedding layer
        self.position_embedding = keras.layers.Embedding(
            input_dim=max_sequence_length,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            dtype=dtype,
            name="position_embedding",
        )

        # Token type embedding layer
        self.token_type_embedding = keras.layers.Embedding(
            input_dim=type_vocab_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            dtype=dtype,
            name="token_type_embedding",
        )

        # Spatial embedding layers for bounding box coordinates
        self.spatial_embeddings = {
            "x": keras.layers.Embedding(
                input_dim=1024,  # Max coordinate value
                output_dim=self.spatial_embedding_dim,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_x_embedding",
            ),
            "y": keras.layers.Embedding(
                input_dim=1024,
                output_dim=self.spatial_embedding_dim,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_y_embedding",
            ),
            "h": keras.layers.Embedding(
                input_dim=1024,
                output_dim=self.spatial_embedding_dim,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_h_embedding",
            ),
            "w": keras.layers.Embedding(
                input_dim=1024,
                output_dim=self.spatial_embedding_dim,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_w_embedding",
            ),
        }

        # Spatial projection layers
        self.spatial_projections = {
            "x": keras.layers.Dense(
                hidden_dim,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_x_projection",
            ),
            "y": keras.layers.Dense(
                hidden_dim,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_y_projection",
            ),
            "h": keras.layers.Dense(
                hidden_dim,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_h_projection",
            ),
            "w": keras.layers.Dense(
                hidden_dim,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name="spatial_w_projection",
            ),
        }

        # Layer normalization and dropout
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="embeddings_layer_norm",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )

        # Transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            layer = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        # Image processing layers
        self.patch_embedding = keras.layers.Conv2D(
            filters=hidden_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="valid",
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            dtype=dtype,
            name="patch_embedding",
        )

        self.patch_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="patch_layer_norm",
        )

        # Initialize the parent class
        super().__init__(dtype=dtype, **kwargs)

    @property
    def token_embedding_matrix(self):
        """Get the token embedding matrix."""
        # Build the layer if not already built
        if not self.token_embedding.built:
            self.token_embedding.build((None, None))
        return self.token_embedding.weights[0]

    def call(self, inputs, training=None):
        """Call the model on new inputs.

        Args:
            inputs: A dictionary containing:
                - "token_ids": Token IDs tensor of shape (batch_size, seq_len)
                - "padding_mask": Padding mask tensor of shape (batch_size, seq_len)
                - "bbox": Bounding box tensor of shape (batch_size, seq_len, 4)
            training: Whether the model is in training mode.

        Returns:
            A tensor of shape (batch_size, seq_len, hidden_dim) containing the
            contextualized representations.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        bbox = inputs["bbox"]

        # Compute sequence length for position embeddings
        seq_length = ops.shape(token_ids)[1]
        batch_size = ops.shape(token_ids)[0]
        # Create position IDs that match the actual sequence length
        position_ids = ops.arange(seq_length, dtype="int32")
        position_ids = ops.expand_dims(position_ids, axis=0)
        # Broadcast to match batch size
        position_ids = ops.tile(position_ids, [batch_size, 1])

        # Token embeddings
        token_embeddings = self.token_embedding(token_ids)

        # Position embeddings
        position_embeddings = self.position_embedding(position_ids)

        # Spatial embeddings
        x_embeddings = self.spatial_embeddings["x"](bbox[..., 0])
        y_embeddings = self.spatial_embeddings["y"](bbox[..., 1])
        h_embeddings = self.spatial_embeddings["h"](bbox[..., 2])
        w_embeddings = self.spatial_embeddings["w"](bbox[..., 3])

        # Project spatial embeddings
        x_embeddings = self.spatial_projections["x"](x_embeddings)
        y_embeddings = self.spatial_projections["y"](y_embeddings)
        h_embeddings = self.spatial_projections["h"](h_embeddings)
        w_embeddings = self.spatial_projections["w"](w_embeddings)

        # Token type embeddings (default to 0)
        token_type_ids = ops.zeros_like(token_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)

        # Combine all embeddings
        embeddings = (
            token_embeddings
            + position_embeddings
            + x_embeddings
            + y_embeddings
            + h_embeddings
            + w_embeddings
            + token_type_embeddings
        )

        # Apply layer normalization and dropout
        embeddings = self.embeddings_layer_norm(embeddings)
        embeddings = self.embeddings_dropout(embeddings, training=training)

        # Apply transformer layers
        hidden_states = embeddings
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(
                hidden_states, padding_mask=padding_mask, training=training
            )

        return hidden_states

    def get_config(self):
        """Get the configuration of the model."""
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "type_vocab_size": self.type_vocab_size,
                "initializer_range": self.initializer_range,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "num_channels": self.num_channels,
                "spatial_embedding_dim": self.spatial_embedding_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create a model from its configuration."""
        return cls(**config)