import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deberta_v3.disentangled_attention_encoder import (
    DisentangledAttentionEncoder,
)
from keras_hub.src.models.deberta_v3.relative_embedding import RelativeEmbedding


def deberta_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.DebertaV3Backbone")
class DebertaV3Backbone(Backbone):
    """DeBERTa encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543).
    It includes the embedding lookups and transformer layers, but does not
    include the enhanced masked decoding head used during pretraining.

    The default constructor gives a fully customizable, randomly initialized
    DeBERTa encoder with any number of layers, heads, and embedding
    dimensions. To load preset architectures and weights, use the `from_preset`
    constructor.

    Note: `DebertaV3Backbone` has a performance issue on TPUs, and we recommend
    other models for TPU training and inference.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/microsoft/DeBERTa).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the DeBERTa model.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length`.
        bucket_size: int. The size of the relative position buckets. Generally
            equal to `max_sequence_length // 2`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Example:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained DeBERTa encoder.
    model = keras_hub.models.DebertaV3Backbone.from_preset(
        "deberta_v3_base_en",
    )
    model(input_data)

    # Randomly initialized DeBERTa encoder with custom config
    model = keras_hub.models.DebertaV3Backbone(
        vocabulary_size=128100,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        intermediate_dim=1536,
        max_sequence_length=512,
        bucket_size=256,
    )
    # Call the model on the input data.
    model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        bucket_size=256,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=deberta_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-7,
            dtype=dtype,
            name="embeddings_layer_norm",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        self.relative_embeddings = RelativeEmbedding(
            hidden_dim=hidden_dim,
            bucket_size=bucket_size,
            layer_norm_epsilon=1e-7,
            kernel_initializer=deberta_kernel_initializer(),
            dtype=dtype,
            name="rel_embedding",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = DisentangledAttentionEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                max_position_embeddings=max_sequence_length,
                bucket_size=bucket_size,
                dropout=dropout,
                activation=keras.activations.gelu,
                layer_norm_epsilon=1e-7,
                kernel_initializer=deberta_kernel_initializer(),
                dtype=dtype,
                name=f"disentangled_attention_encoder_layer_{i}",
            )
            self.transformer_layers.append(layer)

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        rel_embeddings = self.relative_embeddings(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                rel_embeddings=rel_embeddings,
                padding_mask=padding_mask_input,
            )
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.bucket_size = bucket_size
        self.start_token_index = 0

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "bucket_size": self.bucket_size,
            }
        )
        return config
