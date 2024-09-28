import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.f_net_encoder import FNetEncoder
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import gelu_approximate


def f_net_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


def f_net_bias_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.FNetBackbone")
class FNetBackbone(Backbone):
    """A FNet encoder network.

    This class implements a bi-directional Fourier Transform-based encoder as
    described in ["FNet: Mixing Tokens with Fourier Transforms"](https://arxiv.org/abs/2105.03824).
    It includes the embedding lookups and `keras_hub.layers.FNetEncoder` layers,
    but not the masked language model or next sentence prediction heads.

    The default constructor gives a fully customizable, randomly initialized
    FNet encoder with any number of layers and embedding dimensions. To
    load preset architectures and weights, use the `from_preset()` constructor.

    Note: unlike other models, FNet does not take in a `"padding_mask"` input,
    the `"<pad>"` token is handled equivalently to all other tokens in the input
    sequence.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of FNet layers.
        hidden_dim: int. The size of the FNet encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each FNet layer.
        dropout: float. Dropout probability for the embeddings and FNet encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        num_segments: int. The number of types that the 'segment_ids' input can
            take.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained BERT encoder.
    model = keras_hub.models.FNetBackbone.from_preset("f_net_base_en")
    model(input_data)

    # Randomly initialized FNet encoder with a custom config.
    model = keras_hub.models.FNetBackbone(
        vocabulary_size=32000,
        num_layers=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=4,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=f_net_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.position_embedding = PositionEmbedding(
            initializer=f_net_kernel_initializer(),
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="position_embedding",
        )
        self.segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_dim,
            embeddings_initializer=f_net_kernel_initializer(),
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
        self.embedding_projection = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=f_net_kernel_initializer(),
            bias_initializer=f_net_bias_initializer(),
            dtype=dtype,
            name="embedding_projection",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = FNetEncoder(
                intermediate_dim=intermediate_dim,
                activation=gelu_approximate,
                dropout=dropout,
                layer_norm_epsilon=1e-12,
                kernel_initializer=f_net_kernel_initializer(),
                bias_initializer=f_net_bias_initializer(),
                dtype=dtype,
                name=f"f_net_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.pooled_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=f_net_kernel_initializer(),
            bias_initializer=f_net_bias_initializer(),
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
        # Embed tokens, positions, and segment ids.
        tokens = self.token_embedding(token_id_input)
        positions = self.position_embedding(tokens)
        segments = self.segment_embedding(segment_id_input)
        # Sum, normalize and apply dropout to embeddings.
        x = self.embeddings_add((tokens, positions, segments))
        x = self.embeddings_layer_norm(x)
        x = self.embedding_projection(x)
        x = self.embeddings_dropout(x)
        # Apply successive FNet encoder blocks.
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        # Index of classification token in the vocabulary
        cls_token_index = 0
        # Construct the two FNet outputs. The pooled output is a dense layer on
        # top of the [CLS] token.
        sequence_output = x
        pooled_output = self.pooled_dense(x[:, cls_token_index, :])
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "segment_ids": segment_id_input,
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
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.cls_token_index = cls_token_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
            }
        )
        return config
