import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mpnet.mpnet_layers import MPNetRelativePositionBias
from keras_hub.src.models.mpnet.mpnet_layers import MPNetTransformerLayer


def mpnet_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.MPNetBackbone")
class MPNetBackbone(Backbone):
    """An MPNet encoder network.

    This network implements the bi-directional Transformer encoder described
    in ["MPNet: Masked and Permuted Pre-training for Language Understanding"](https://arxiv.org/abs/2004.09297).
    It includes the embedding lookups and transformer layers, but does not
    include the masked and permuted language modeling head used during
    pretraining.

    MPNet combines learned absolute position embeddings with a bucketed
    relative position bias, which is shared by all transformer layers.

    The default constructor gives a fully customizable, randomly initialized
    MPNet encoder with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset()`
    constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/microsoft/MPNet).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the embeddings and the
            Transformer encoder. Defaults to `0.1`.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. This determines the variable shape for the absolute
            position embeddings. Defaults to `512`.
        layer_norm_epsilon: float. The epsilon of the layer normalization
            layers. Defaults to `1e-5`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "padding_mask": np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype="int32"
        ),
    }

    # Pretrained MPNet encoder.
    model = keras_hub.models.MPNetBackbone.from_preset("all_mpnet_base_v2_en")
    model(input_data)

    # Randomly initialized MPNet encoder with a custom config.
    model = keras_hub.models.MPNetBackbone(
        vocabulary_size=30527,
        num_layers=4,
        num_heads=4,
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
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        # `mask_zero` must stay `False`: id 0 is the `<s>` token, and padding
        # is passed explicitly through `padding_mask`.
        self.embeddings = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=mpnet_kernel_initializer(),
            dtype=dtype,
            name="embeddings",
        )
        self.token_embedding = self.embeddings.token_embedding
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="embeddings_layer_norm",
        )
        self.embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        self.relative_position_bias = MPNetRelativePositionBias(
            num_heads=num_heads,
            initializer=mpnet_kernel_initializer(),
            dtype=dtype,
            name="relative_position_bias",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = MPNetTransformerLayer(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                activation="gelu",
                layer_norm_epsilon=layer_norm_epsilon,
                kernel_initializer=mpnet_kernel_initializer(),
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
        x = self.embeddings(token_id_input)
        x = self.embeddings_layer_norm(x)
        x = self.embeddings_dropout(x)
        # The relative position bias is computed once and shared by all
        # transformer layers.
        position_bias = self.relative_position_bias(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(
                x,
                padding_mask=padding_mask_input,
                position_bias=position_bias,
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
        self.layer_norm_epsilon = layer_norm_epsilon

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
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
