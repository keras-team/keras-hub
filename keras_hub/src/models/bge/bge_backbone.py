import keras
from keras.layers import ReversibleEmbedding

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.models.backbone import Backbone


def bge_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.BgeBackbone")
class BgeBackbone(Backbone):
    """A BGE (BAAI General Embedding) encoder backbone.

    This class implements a bi-directional Transformer-based encoder following
    the architecture of `BAAI/bge-*-en-v1.5` models. The BGE model family
    uses a standard BERT-style encoder (word + position + segment embeddings,
    multiple `TransformerEncoder` layers, and a CLS-token pooler) and is
    fine-tuned with contrastive learning for dense text retrieval and semantic
    similarity tasks.

    The default constructor gives a fully customizable, randomly initialized
    BGE encoder. To load preset architectures and weights, use the
    `from_preset()` constructor.

    Note: For sentence embeddings, extract the `[CLS]` token's hidden state
    from `sequence_output` and apply L2 normalization. Do not use
    `pooled_output` for similarity tasks—it passes through a Tanh activation
    and produces embeddings in a different space.

    ```python
    import keras
    backbone = keras_hub.models.BgeBackbone.from_preset("bge_small_en_v1.5")
    outputs = backbone(inputs)
    cls_emb = outputs["sequence_output"][:, 0, :]
    emb = keras.ops.normalize(cls_emb, axis=-1)  # L2-normalized embedding
    ```

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://huggingface.co/BAAI/bge-small-en-v1.5).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer encoder layers.
        num_heads: int. The number of attention heads for each transformer.
            The `hidden_dim` must be divisible by `num_heads`.
        hidden_dim: int. The hidden dimension of the transformer encoder and
            pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the two-layer feedforward network for each transformer encoder.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. Determines the shape of the learned positional embeddings.
        num_segments: int. The number of distinct segment types (used for
            `segment_ids` input). Typically 2 for sentence-pair encoding.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "segment_ids": np.zeros(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained BGE encoder.
    model = keras_hub.models.BgeBackbone.from_preset("bge_small_en_v1.5")
    model(input_data)

    # Randomly initialized BGE encoder with a custom config.
    model = keras_hub.models.BgeBackbone(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=384,
        intermediate_dim=1536,
        max_sequence_length=512,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=bge_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.position_embedding = PositionEmbedding(
            initializer=bge_kernel_initializer(),
            sequence_length=max_sequence_length,
            dtype=dtype,
            name="position_embedding",
        )
        self.segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_dim,
            embeddings_initializer=bge_kernel_initializer(),
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
        self.transformer_layers = []
        for i in range(num_layers):
            layer = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation="gelu",
                dropout=dropout,
                layer_norm_epsilon=1e-12,
                kernel_initializer=bge_kernel_initializer(),
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.pooled_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=bge_kernel_initializer(),
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
        # Embed tokens, positions, and segment ids then sum and normalize.
        tokens = self.token_embedding(token_id_input)
        positions = self.position_embedding(tokens)
        segments = self.segment_embedding(segment_id_input)
        x = self.embeddings_add((tokens, positions, segments))
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask_input)
        # sequence_output: full hidden states [B, L, hidden_dim].
        # pooled_output: Tanh-activated CLS representation [B, hidden_dim].
        # NOTE: For BGE sentence embeddings, use sequence_output[:, 0, :]
        # (the raw CLS token) followed by L2 normalization—do not use
        # pooled_output for similarity tasks.
        sequence_output = x
        cls_token_index = 0
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
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
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
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
            }
        )
        return config
