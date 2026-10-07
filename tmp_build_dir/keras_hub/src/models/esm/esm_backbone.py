import keras
from keras import activations

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.esm.esm_encoder import ESMEncoder


def esm2_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export(
    ["keras_hub.models.ESM2Backbone", "keras_hub.models.ESMBackbone"]
)
class ESMBackbone(Backbone):
    """A ESM2 and ESM encoder network.

    This class implements a bi-directional Transformer-based encoder as
    described in ["ESM"](https://github.com/facebookresearch/esm).

    The default constructor gives a fully customizable, randomly initialized
    ESM2 encoder with any number of layers, heads, and embed dim.To
    load preset architectures and weights, use the `from_preset()` constructor.


    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
            Defaults to 0.1
        use_pre_layer_norm:bool.If true, then layer norm will be used before
            entering the transformer block.
            Since it's pre-norm, the default is false.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        position_embedding_type: str. The position embedding type to use.
            One of "absolute" and "rotary".
            Use "absolute" for ESM1. Use "rotary" for ESM2. Defaults to "rotary"
        max_wavelength : int. The maximum  angular wavelength of
            the sine/cosine curves, for rotary embeddings.
            Defaults to `10000`.
        activation :string or keras.activations. The activation to
            use for the transformer.
            Defaults to `"gelu"`.
        pad_token_id: int.padding token id. Normally 0,
            but is set to 1 in the esm2 model.
            Defaults to 0.
        dtype: None or str or keras.mixed_precision.DTypePolicy. The dtype to
            use for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    }

    # Pretrained ESM2 encoder.
    model = keras_hub.models.ESM2Backbone.from_preset('hf://facebook/esm2_t6_8M_UR50D')
    model(input_data)

    # Randomly initialized ESM2 encoder with a custom config.
    model = keras_hub.models.ESM2Backbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
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
        use_bias=True,
        activation="gelu",
        dropout=0.1,
        dtype=None,
        max_sequence_length=1024,
        max_wavelength=10000,
        layer_norm_eps=1e-12,
        use_pre_layer_norm=False,
        position_embedding_type="rotary",
        pad_token_id=0,
        **kwargs,
    ):
        if position_embedding_type not in (
            "rotary",
            "absolute",
        ):
            raise ValueError(
                '`position_embedding_type` must be either `"rotary"`, or '
                '`"absolute"`. Received '
                f"position_embedding_type={position_embedding_type}."
            )
        head_size = hidden_dim // num_heads
        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=esm2_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )
        if position_embedding_type == "absolute":
            self.position_embedding = PositionEmbedding(
                initializer=esm2_kernel_initializer(),
                sequence_length=max_sequence_length,
                dtype=dtype,
                name="position_embedding",
            )
            self.embeddings_add = keras.layers.Add(
                dtype=dtype,
                name="embeddings_add",
            )

        self.output_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            dtype=dtype,
            name="output_layer_norm",
        )
        if use_pre_layer_norm:
            self.emb_layer_norm = keras.layers.LayerNormalization(
                epsilon=layer_norm_eps,
                dtype=dtype,
                name="emb_layer_norm",
            )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = ESMEncoder(
                heads=num_heads,
                head_size=head_size,
                intermediate_size=intermediate_dim,
                use_bias=use_bias,
                max_wavelength=max_wavelength,
                dropout=dropout,
                activation=activation,
                kernel_initializer=esm2_kernel_initializer(),
                layer_norm_eps=layer_norm_eps,
                dtype=dtype,
                use_rotary=position_embedding_type == "rotary",
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )

        attention_mask = keras.ops.not_equal(token_id_input, pad_token_id)

        token_vector = self.token_embedding(token_id_input)
        if position_embedding_type == "absolute":
            position_vector = self.position_embedding(
                token_vector, start_index=pad_token_id
            )
            x = self.embeddings_add([token_vector, position_vector])
        else:
            x = token_vector
        if use_pre_layer_norm:
            x = self.emb_layer_norm(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=attention_mask)
        output = self.output_layer_norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
            },
            outputs=output,
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
        self.max_wavelength = max_wavelength
        self.head_size = head_size
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.start_token_index = 0
        self.layer_norm_eps = layer_norm_eps
        self.max_sequence_length = max_sequence_length
        self.use_pre_layer_norm = use_pre_layer_norm
        self.position_embedding_type = position_embedding_type
        self.pad_token_id = pad_token_id

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
                "max_wavelength": self.max_wavelength,
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
                "layer_norm_eps": self.layer_norm_eps,
                "use_pre_layer_norm": self.use_pre_layer_norm,
                "position_embedding_type": self.position_embedding_type,
                "max_sequence_length": self.max_sequence_length,
                "pad_token_id": self.pad_token_id,
            }
        )
        return config
