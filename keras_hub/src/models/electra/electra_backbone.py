import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import gelu_approximate


def electra_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.ElectraBackbone")
class ElectraBackbone(Backbone):
    """A Electra encoder network.

    This network implements a bidirectional Transformer-based encoder as
    described in ["Electra: Pre-training Text Encoders as Discriminators Rather
    Than Generators"](https://arxiv.org/abs/2003.10555). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default constructor gives a fully customizable, randomly initialized
    ELECTRA encoder with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the
    `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://huggingface.co/docs/transformers/model_doc/electra#overview).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        embedding_dim: int. The size of the token embeddings.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Example:
        ```python
        input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
        }

        # Pre-trained ELECTRA encoder.
        model = keras_hub.models.ElectraBackbone.from_preset(
            "electra_base_discriminator_en"
        )
        model(input_data)

        # Randomly initialized Electra encoder
        backbone = keras_hub.models.ElectraBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            intermediate_dim=64,
            dropout=0.1,
            max_sequence_length=512,
            )
        # Returns sequence and pooled outputs.
        sequence_output, pooled_output = backbone(input_data)
        ```
    """

    def __init__(
        self,
        vocab_size,
        num_layers,
        num_heads,
        hidden_dim,
        embedding_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=electra_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.position_embedding = PositionEmbedding(
            initializer=electra_kernel_initializer(),
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="position_embedding",
        )
        self.segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=embedding_dim,
            embeddings_initializer=electra_kernel_initializer(),
            dtype=dtype,
            name="segment_embedding",
        )
        self.embeddings_add = keras.layers.Add(
            dtype=dtype,
            name="embeddings_add",
        )
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-12,
            dtype=dtype,
            name="embeddings_layer_norm",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        if hidden_dim != embedding_dim:
            self.embeddings_projection = keras.layers.Dense(
                hidden_dim,
                kernel_initializer=electra_kernel_initializer(),
                dtype=dtype,
                name="embeddings_projection",
            )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=gelu_approximate,
                dropout=dropout,
                layer_norm_epsilon=1e-12,
                kernel_initializer=electra_kernel_initializer(),
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.pooled_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=electra_kernel_initializer(),
            activation="tanh",
            dtype=dtype,
            name="pooled_dense",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        # Embed tokens, positions, and segment ids.
        tokens = self.token_embedding(token_id_input)
        positions = self.position_embedding(tokens)
        segments = self.segment_embedding(segment_id_input)
        # Add all embeddings together.
        x = self.embeddings_add((tokens, positions, segments))
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        if hidden_dim != embedding_dim:
            x = self.embeddings_projection(x)
        # Apply successive transformer encoder blocks.
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
        # Index of classification token in the vocabulary
        cls_token_index = 0
        sequence_output = x
        # Construct the two ELECTRA outputs. The pooled output is a dense layer
        # on top of the [CLS] token.
        pooled_output = self.pooled_dense(x[:, cls_token_index, :])
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "segment_ids": segment_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.cls_token_index = cls_token_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
            }
        )
        return config
