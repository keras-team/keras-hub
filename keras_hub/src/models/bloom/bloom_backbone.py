import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bloom.bloom_decoder import BloomDecoder


def _bloom_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.BloomBackbone")
class BloomBackbone(Backbone):
    """A BLOOM decoder network.

    This network implements a Transformer-based decoder network, BigScience
    Language Open-science Open-access Multilingual (BLOOM), as descriped in
    ["BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"](https://arxiv.org/pdf/2211.05100.pdf).

    The default constructor gives a fully customizable, randomly initialized
    Bloom model with any number of layers, heads, and embedding dimensions. To
    load preset architectures and weights, use the `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available [here](https://huggingface.co/spaces/bigscience/license).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The dimensionality of the embeddings and hidden states.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the MLP network of each transformer.
        dropout: float. Dropout probability for the Transformer decoder.
        layer_norm_epsilon: float. Epsilon for the layer normalization layers in
            the transformer decoder.
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

    # Pretrained BLOOM decoder.
    model = keras_hub.models.BloomBackbone.from_preset("bloom_560m_multi")
    model(input_data)

    # Randomly initialized BLOOM decoder with a custom config.
    model = keras_hub.models.BloomBackbone(
        vocabulary_size=10,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        intermediate_dim=32*4,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
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
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_bloom_kernel_initializer(stddev=0.02),
            dtype=dtype,
            name="token_embedding",
        )
        self.embeddings_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="embedding_layernorm",
        )
        self.transformer_layers = []
        for i in range(num_layers):
            layer = BloomDecoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        x = self.token_embedding(token_id_input)
        x = self.embeddings_layer_norm(x)
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, decoder_padding_mask=padding_mask_input)
        sequence_output = self.layer_norm(x)
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
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
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
