import keras
from keras import activations

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.roformer_v2.roformer_v2_attention import RoformerNorm
from keras_hub.src.models.roformer_v2.roformer_v2_encoder import (
    RoformerV2Encoder,
)


def roformer_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.RoformerV2Backbone")
class RoformerV2Backbone(Backbone):
    """A RoformerV2 encoder network.

    This class implements a bi-directional Transformer-based encoder as
    described in ["Roformer"](https://github.com/ZhuiyiTechnology/roformer).
    It includes the
    embedding lookups and transformer layers, but not the masked language model
    or next sentence prediction heads.

    The default constructor gives a fully customizable, randomly initialized
    RoformerV2 encoder with any number of layers, heads, and embed dim.To
    load preset architectures and weights, use the `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
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
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    # Pretrained RoformerV2 encoder.
    model = keras_hub.models.RoformerV2Backbone.from_preset("roformer_v2_base")
    model(input_data)

    # Randomly initialized RoformerV2 encoder with a custom config.
    model = keras_hub.models.RoformerV2Backbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        head_size = 64,
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
        head_size,
        use_bias=False,
        activation="relu",
        dropout=0.1,
        num_segments=2,
        dtype=None,
        max_wavelength=10000,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=roformer_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        self.segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_dim,
            embeddings_initializer=roformer_kernel_initializer(),
            dtype=dtype,
            name="segment_embedding",
        )
        self.embeddings_add = keras.layers.Add(
            dtype=dtype,
            name="embeddings_add",
        )
        self.embeddings_layer_norm = RoformerNorm(
            epsilon=keras.backend.epsilon(),
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
            layer = RoformerV2Encoder(
                heads=num_heads,
                head_size=head_size,
                intermediate_size=intermediate_dim,
                use_bias=use_bias,
                max_wavelength=max_wavelength,
                dropout=dropout,
                activation=activation,
                kernel_initializer=roformer_kernel_initializer(),
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        attention_mask = keras.ops.not_equal(token_id_input, 0)
        # Embed tokens, positions, and segment ids.
        tokens = self.token_embedding(token_id_input)
        segments = self.segment_embedding(segment_id_input)
        # Sum, normalize and apply dropout to embeddings.
        x = self.embeddings_add((tokens, segments))
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=attention_mask)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "segment_ids": segment_id_input,
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
        self.num_segments = num_segments
        self.max_wavelength = max_wavelength
        self.head_size = head_size
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.use_bias = use_bias
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
                "num_segments": self.num_segments,
                "max_wavelength": self.max_wavelength,
                "head_size": self.head_size,
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
            }
        )
        return config
