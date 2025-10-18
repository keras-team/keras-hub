import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.layoutlmv3.layoutlmv3_transformer import (
    LayoutLMv3TransformerLayer,
)


@keras_hub_export("keras_hub.models.LayoutLMv3Backbone")
class LayoutLMv3Backbone(Backbone):
    """LayoutLMv3 backbone model for document understanding tasks.

    This class implements the LayoutLMv3 model architecture for joint text and
    layout understanding in document AI tasks. It processes both text and image
    inputs while maintaining spatial relationships in documents.

    The default constructor gives a fully customizable, randomly initialized
    LayoutLMv3 model with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset` constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary. Defaults to 
            30522.
        hidden_dim: int. The size of the transformer hidden state at the end of
            each transformer layer. Defaults to 768.
        num_layers: int. The number of transformer layers. Defaults to 12.
        num_heads: int. The number of attention heads for each transformer.
            Defaults to 12.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer. Defaults to
            3072.
        dropout: float. Dropout probability for the transformer encoder.
            Defaults to 0.1.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. Defaults to 512.
        type_vocab_size: int. The vocabulary size for token types. Defaults to 
            2.
        initializer_range: float. The standard deviation of the truncated_normal
            initializer for initializing all weight matrices. Defaults to 0.02.
        layer_norm_epsilon: float. The epsilon used by the layer normalization
            layers. Defaults to 1e-12.
        spatial_embedding_dim: int. The dimension of spatial position 
            embeddings for bounding box coordinates. Defaults to 64.
        patch_size: int. The size of the patches for image processing. Defaults
            to 16.
        num_channels: int. The number of channels in the input images. Defaults
            to 3.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
        "bbox": np.ones(shape=(1, 12, 4), dtype="int32"),
    }

    # Pretrained LayoutLMv3 encoder.
    model = keras_hub.models.LayoutLMv3Backbone.from_preset(
        "layoutlmv3_base",
    )
    model(input_data)

    # Randomly initialized LayoutLMv3 encoder with custom config.
    model = keras_hub.models.LayoutLMv3Backbone(
        vocabulary_size=30522,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        max_sequence_length=512,
        spatial_embedding_dim=64,
    )
    model(input_data)
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
        spatial_embedding_dim=64,
        patch_size=16,
        num_channels=3,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            dtype=dtype,
            name="token_embedding",
        )

        self.position_embedding = keras.layers.Embedding(
            input_dim=max_sequence_length,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            dtype=dtype,
            name="position_embedding",
        )

        # Spatial position embeddings for bounding box coordinates
        self.spatial_embeddings = {}
        self.spatial_projections = {}
        for coord in ["x", "y", "h", "w"]:
            self.spatial_embeddings[coord] = keras.layers.Embedding(
                input_dim=1024,
                output_dim=spatial_embedding_dim,
                embeddings_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name=f"{coord}_position_embedding",
            )
            self.spatial_projections[coord] = keras.layers.Dense(
                hidden_dim,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
                dtype=dtype,
                name=f"{coord}_projection",
            )

        self.token_type_embedding = keras.layers.Embedding(
            input_dim=type_vocab_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=initializer_range
            ),
            dtype=dtype,
            name="token_type_embedding",
        )

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
            layer = LayoutLMv3TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation="gelu",
                layer_norm_epsilon=layer_norm_epsilon,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=initializer_range
                ),
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

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        bbox_input = keras.Input(
            shape=(None, 4), dtype="int32", name="bbox"
        )

        # Compute sequence length for position embeddings
        seq_length = ops.shape(token_id_input)[1]
        position_ids = ops.arange(seq_length, dtype="int32")
        position_ids = ops.expand_dims(position_ids, axis=0)
        position_ids = ops.broadcast_to(
            position_ids, ops.shape(token_id_input)
        )

        # Token embeddings
        token_embeddings = self.token_embedding(token_id_input)
        
        # Position embeddings
        position_embeddings = self.position_embedding(position_ids)

        # Spatial embeddings
        x_embeddings = self.spatial_embeddings["x"](bbox_input[..., 0])
        y_embeddings = self.spatial_embeddings["y"](bbox_input[..., 1])
        h_embeddings = self.spatial_embeddings["h"](bbox_input[..., 2])
        w_embeddings = self.spatial_embeddings["w"](bbox_input[..., 3])

        # Project spatial embeddings
        x_embeddings = self.spatial_projections["x"](x_embeddings)
        y_embeddings = self.spatial_projections["y"](y_embeddings)
        h_embeddings = self.spatial_projections["h"](h_embeddings)
        w_embeddings = self.spatial_projections["w"](w_embeddings)

        # Token type embeddings (default to 0)
        token_type_ids = ops.zeros_like(token_id_input)
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
        embeddings = self.embeddings_dropout(embeddings)

        # Apply transformer layers
        hidden_states = embeddings
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(
                hidden_states, padding_mask=padding_mask_input
            )

        # Build the model
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
                "bbox": bbox_input,
            },
            outputs=hidden_states,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
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
        self.spatial_embedding_dim = spatial_embedding_dim
        self.patch_size = patch_size
        self.num_channels = num_channels

    def get_config(self):
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
                "spatial_embedding_dim": self.spatial_embedding_dim,
                "patch_size": self.patch_size,
                "num_channels": self.num_channels,
            }
        )
        return config

    @property
    def token_embedding_matrix(self):
        return self.token_embedding.embeddings
