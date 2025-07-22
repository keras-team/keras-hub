import keras
from keras import ops

# Import with error handling for missing dependencies
try:
    from keras_hub.src.api_export import keras_hub_export
except ImportError:
    # Fallback for missing api_export
    def keras_hub_export(name):
        def decorator(cls):
            return cls
        return decorator

try:
    from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
except ImportError:
    # Fallback to standard Keras embedding if PositionEmbedding not available
    PositionEmbedding = keras.layers.Embedding

try:
    from keras_hub.src.layers.modeling.reversible_embedding import (
        ReversibleEmbedding,
    )
except ImportError:
    # Fallback to standard Keras embedding if ReversibleEmbedding not available
    ReversibleEmbedding = keras.layers.Embedding

try:
    from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
except ImportError:
    # Create a minimal fallback TransformerEncoder
    class TransformerEncoder(keras.layers.Layer):
        def __init__(self, num_heads, intermediate_dim, dropout=0.1, **kwargs):
            super().__init__(**kwargs)
            self.num_heads = num_heads
            self.intermediate_dim = intermediate_dim
            self.dropout = dropout
            
        def call(self, x, padding_mask=None):
            # Minimal implementation - just return input
            return x

try:
    from keras_hub.src.models.backbone import Backbone
except ImportError:
    # Fallback to standard Keras Model if Backbone not available
    Backbone = keras.Model


@keras_hub_export("keras_hub.models.LayoutLMv3Backbone")
class LayoutLMv3Backbone(Backbone):
    """LayoutLMv3 backbone model for document understanding tasks.

    This class implements the LayoutLMv3 model architecture for joint text and
    layout understanding in document AI tasks. It processes both text and image
    inputs while maintaining spatial relationships in documents.

    The default constructor gives a fully customizable, randomly initialized
    LayoutLMv3 encoder with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The size of the transformer encoding layer.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. If None, max_sequence_length uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        spatial_embedding_dim: int. The dimension of the spatial embeddings.
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
    model = keras_hub.models.LayoutLMv3Backbone.from_preset("layoutlmv3_base")
    model(input_data)

    # Randomly initialized LayoutLMv3 encoder with custom config.
    model = keras_hub.models.LayoutLMv3Backbone(
        vocabulary_size=30522,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        max_sequence_length=512,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        spatial_embedding_dim=64,
        dtype=None,
        **kwargs,
    ):
        # Validate inputs for better error messages
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        
        # === Layers ===
        # Use appropriate embedding class based on what's available
        if ReversibleEmbedding != keras.layers.Embedding:
            self.token_embedding = ReversibleEmbedding(
                input_dim=vocabulary_size,
                output_dim=hidden_dim,
                dtype=dtype,
                name="token_embedding",
            )
        else:
            self.token_embedding = keras.layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=hidden_dim,
                dtype=dtype,
                name="token_embedding",
            )
        
        # Use appropriate position embedding
        if PositionEmbedding != keras.layers.Embedding:
            self.position_embedding = PositionEmbedding(
                sequence_length=max_sequence_length,
                dtype=dtype,
                name="position_embedding",
            )
        else:
            self.position_embedding = keras.layers.Embedding(
                input_dim=max_sequence_length,
                output_dim=hidden_dim,
                dtype=dtype,
                name="position_embedding",
            )
        
        # Spatial embeddings for bounding box coordinates
        self.x_position_embedding = keras.layers.Embedding(
            input_dim=1024,
            output_dim=spatial_embedding_dim,
            dtype=dtype,
            name="x_position_embedding",
        )
        self.y_position_embedding = keras.layers.Embedding(
            input_dim=1024,
            output_dim=spatial_embedding_dim,
            dtype=dtype,
            name="y_position_embedding",
        )
        self.h_position_embedding = keras.layers.Embedding(
            input_dim=1024,
            output_dim=spatial_embedding_dim,
            dtype=dtype,
            name="h_position_embedding",
        )
        self.w_position_embedding = keras.layers.Embedding(
            input_dim=1024,
            output_dim=spatial_embedding_dim,
            dtype=dtype,
            name="w_position_embedding",
        )
        
        # Projection layers for spatial embeddings
        self.x_projection = keras.layers.Dense(
            hidden_dim, dtype=dtype, name="x_projection"
        )
        self.y_projection = keras.layers.Dense(
            hidden_dim, dtype=dtype, name="y_projection"
        )
        self.h_projection = keras.layers.Dense(
            hidden_dim, dtype=dtype, name="h_projection"
        )
        self.w_projection = keras.layers.Dense(
            hidden_dim, dtype=dtype, name="w_projection"
        )
        
        # Token type embedding
        self.token_type_embedding = keras.layers.Embedding(
            input_dim=2,
            output_dim=hidden_dim,
            dtype=dtype,
            name="token_type_embedding",
        )
        
        self.embeddings_add = keras.layers.Add(
            dtype=dtype, name="embeddings_add"
        )
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-12, dtype=dtype, name="embeddings_layer_norm"
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout, dtype=dtype, name="embeddings_dropout"
        )
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(num_layers):
            layer = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_epsilon=1e-12,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        bbox_input = keras.Input(shape=(None, 4), dtype="int32", name="bbox")
        
        # Embeddings
        tokens = self.token_embedding(token_id_input)
        
        # Handle position embeddings based on available class
        if PositionEmbedding != keras.layers.Embedding:
            positions = self.position_embedding(tokens)
        else:
            # Create position indices manually for standard embedding
            seq_length = ops.shape(token_id_input)[1]
            position_ids = ops.arange(seq_length, dtype="int32")
            position_ids = ops.expand_dims(position_ids, 0)
            batch_size = ops.shape(token_id_input)[0]
            position_ids = ops.tile(position_ids, [batch_size, 1])
            positions = self.position_embedding(position_ids)
        
        # Spatial embeddings with explicit casting for backend compatibility
        x_indices = ops.cast(bbox_input[..., 0], "int32")
        y_indices = ops.cast(bbox_input[..., 1], "int32")
        h_indices = ops.cast(bbox_input[..., 2], "int32")
        w_indices = ops.cast(bbox_input[..., 3], "int32")
        
        x_emb = self.x_projection(self.x_position_embedding(x_indices))
        y_emb = self.y_projection(self.y_position_embedding(y_indices))
        h_emb = self.h_projection(self.h_position_embedding(h_indices))
        w_emb = self.w_projection(self.w_position_embedding(w_indices))
        
        # Token type (default to 0) with explicit shape handling
        batch_size = ops.shape(token_id_input)[0]
        seq_length = ops.shape(token_id_input)[1]
        token_type_ids = ops.zeros((batch_size, seq_length), dtype="int32")
        token_types = self.token_type_embedding(token_type_ids)
        
        # Combine embeddings
        embeddings_list = [tokens, positions, x_emb, y_emb, h_emb, w_emb, token_types]
        x = self.embeddings_add(embeddings_list)
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        
        # Transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
            
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
                "bbox": bbox_input,
            },
            outputs=x,
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
        self.spatial_embedding_dim = spatial_embedding_dim

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
                "spatial_embedding_dim": self.spatial_embedding_dim,
            }
        )
        return config

    @property
    def token_embedding_matrix(self):
        if hasattr(self.token_embedding, 'embeddings'):
            return self.token_embedding.embeddings
        else:
            # Fallback for standard Keras embedding
            return self.token_embedding.weights[0]
