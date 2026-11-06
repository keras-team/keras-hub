import keras
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertEncoderLayer,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertTrigramProjection,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import NoNorm


def mobile_bert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.MobileBertBackbone")
class MobileBertBackbone(Backbone):
    """A MobileBERT encoder network.

    This class implements a bi-directional Transformer-based encoder as
    described in ["MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
    Devices"](https://arxiv.org/abs/2004.02984). It includes the
    embedding lookups and transformer layers, but not the masked language model
    head.

    The default constructor gives a fully customizable, randomly initialized
    MobileBERT encoder with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the
    `from_preset()`
    constructor.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        embedding_dim: int. The size of the token embeddings.
        intra_bottleneck_dim: int. The size of the bottleneck layers in the
            transformer blocks.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a feedforward network for each transformer.
        num_feedforward_networks: int. Number of stacked feed-forward networks
            in each transformer layer.
        activation: string or callable. The activation function for the hidden
            layers.
        normalization_type: string. The type of normalization to use. Either
            `"no_norm"` or `"layer_norm"`.
        layer_norm_epsilon: float. Epsilon value for layer normalization.
            Applied to every normalization layer when
            `normalization_type="layer_norm"`. Note that the Hugging Face
            implementation leaves two of its norms (the embedding norm and the
            last feed-forward output norm) at PyTorch's default epsilon of
            1e-5; this port uses `layer_norm_epsilon` uniformly. This has no
            effect for `"no_norm"`, which is what every released MobileBERT
            checkpoint uses.
        use_pooler_dense: boolean. Whether to apply a dense layer to the pooled
            output.
        dropout: float. Dropout probability for the hidden layers.
        attention_dropout: float. Dropout probability for the attention probs.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume.
        num_segments: int. The number of types that the 'segment_ids' input can
            take.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        embedding_dim,
        intra_bottleneck_dim,
        intermediate_dim,
        num_feedforward_networks=4,
        activation="relu",
        normalization_type="no_norm",
        layer_norm_epsilon=1e-12,
        use_pooler_dense=False,
        dropout=0.0,
        attention_dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            embeddings_initializer=mobile_bert_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.embedding_projection = MobileBertTrigramProjection(
            hidden_dim=hidden_dim,
            dtype=dtype,
            name="embedding_projection",
        )
        self.position_embedding = PositionEmbedding(
            initializer=mobile_bert_kernel_initializer(),
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="position_embedding",
        )
        self.segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_dim,
            embeddings_initializer=mobile_bert_kernel_initializer(),
            dtype=dtype,
            name="segment_embedding",
        )
        self.embeddings_add = keras.layers.Add(
            dtype=dtype, name="embeddings_add"
        )
        if normalization_type == "no_norm":
            self.embeddings_layer_norm = NoNorm(
                dtype=dtype, name="embeddings_layer_norm"
            )
        else:
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

        self.transformer_layers = []
        for i in range(num_layers):
            layer = MobileBertEncoderLayer(
                hidden_dim=hidden_dim,
                intra_bottleneck_dim=intra_bottleneck_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                num_feedforward_networks=num_feedforward_networks,
                activation=activation,
                normalization_type=normalization_type,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                attention_dropout=attention_dropout,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        if use_pooler_dense:
            self.pooled_dense = keras.layers.Dense(
                hidden_dim,
                kernel_initializer=mobile_bert_kernel_initializer(),
                activation="tanh",
                dtype=dtype,
                name="pooled_dense",
            )
        else:
            self.pooled_dense = None

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

        # Embeddings.
        x = self.token_embedding(token_id_input)
        x = self.embedding_projection(x)
        position_embeddings = self.position_embedding(x)
        segment_embeddings = self.segment_embedding(segment_id_input)

        x = self.embeddings_add((x, position_embeddings, segment_embeddings))
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)

        sequence_output = x
        if self.pooled_dense is not None:
            pooled_output = self.pooled_dense(sequence_output[:, 0, :])
        else:
            pooled_output = sequence_output[:, 0, :]

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
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.intra_bottleneck_dim = intra_bottleneck_dim
        self.intermediate_dim = intermediate_dim
        self.num_feedforward_networks = num_feedforward_networks
        self.activation = activation
        self.normalization_type = normalization_type
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_pooler_dense = use_pooler_dense
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "embedding_dim": self.embedding_dim,
                "intra_bottleneck_dim": self.intra_bottleneck_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_feedforward_networks": self.num_feedforward_networks,
                "activation": self.activation,
                "normalization_type": self.normalization_type,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "use_pooler_dense": self.use_pooler_dense,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
            }
        )
        return config
