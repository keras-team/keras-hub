import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.whisper.whisper_decoder import WhisperDecoder
from keras_hub.src.models.whisper.whisper_encoder import WhisperEncoder


def whisper_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


class Padder(keras.layers.Layer):
    def call(self, x):
        return ops.pad(x, [[0, 0], [1, 1], [0, 0]])


@keras_hub_export("keras_hub.models.WhisperBackbone")
class WhisperBackbone(Backbone):
    """A Whisper encoder-decoder network for speech.

    This class implements a Transformer-based encoder-decoder model as
    described in
    ["Robust Speech Recognition via Large-Scale Weak Supervision"](https://arxiv.org/abs/2212.04356).
    It includes the embedding lookups and transformer layers, but not the head
    for predicting the next token.

    The default constructor gives a fully customizable, randomly initialized
    Whisper model with any number of layers, heads, and embedding dimensions.
    To load preset architectures and weights, use the `from_preset()`
    constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/whisper).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer encoder layers and
            transformer decoder layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        num_mels: int. The number of mel-frequency filters. Defaults to `80`.
        dropout: float. Dropout probability for the Transformer encoder.
        max_encoder_sequence_length: int. The maximum sequence length that the
            audio encoder can consume. Since the second convolutional layer in
            the encoder reduces the sequence length by half (stride of 2), we
            use `max_encoder_sequence_length // 2` as the sequence length for
            the positional embedding layer.
        max_decoder_sequence_length: int. The maximum sequence length that the
            text decoder can consume.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Note that some computations,
            such as softmax and layer normalization, will always be done at
            float32 precision regardless of dtype.

    Examples:

    ```python
    input_data = {
        "encoder_features": np.ones(shape=(1, 12, 80), dtype="int32"),
        "decoder_token_ids": np.ones(shape=(1, 12), dtype="int32"),
        "decoder_padding_mask": np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
        ),
    }

    # Randomly initialized Whisper encoder-decoder model with a custom config.
    model = keras_hub.models.WhisperBackbone(
        vocabulary_size=51864,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_encoder_sequence_length=128,
        max_decoder_sequence_length=128,
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
        num_mels=80,
        dropout=0.0,
        max_encoder_sequence_length=3000,
        max_decoder_sequence_length=448,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.encoder_conv_layer_1 = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            dtype=dtype,
            name="encoder_token_embedding_conv_layer_1",
        )
        self.encoder_conv_layer_2 = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=3,
            strides=2,
            padding="valid",
            dtype=dtype,
            name="encoder_token_embedding_conv_layer_2",
        )
        self.encoder_padder = Padder(
            dtype=dtype,
            name="encoder_padder",
        )
        self.encoder_position_embedding = PositionEmbedding(
            initializer=whisper_kernel_initializer(),
            sequence_length=max_encoder_sequence_length // 2,
            dtype=dtype,
            name="encoder_position_embedding",
            trainable=False,
        )
        self.encoder_embeddings_add = keras.layers.Add(
            dtype=dtype,
            name="encoder_embeddings_add",
        )
        self.encoder_embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_embeddings_dropout",
        )
        self.encoder_transformer_layers = []
        for i in range(num_layers):
            layer = WhisperEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=keras.activations.gelu,
                layer_norm_epsilon=1e-5,
                dropout=dropout,
                kernel_initializer=whisper_kernel_initializer(),
                normalize_first=True,
                dtype=dtype,
                name=f"transformer_encoder_layer_{i}",
            )
            self.encoder_transformer_layers.append(layer)
        self.encoder_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=dtype,
            name="encoder_layer_norm",
        )
        self.decoder_embeddings = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_decoder_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=whisper_kernel_initializer(),
            dtype=dtype,
            name="decoder_token_and_position_embedding",
        )
        self.token_embedding = self.decoder_embeddings.token_embedding
        self.decoder_embeddings_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="decoder_embeddings_dropout",
        )
        self.decoder_transformer_layers = []
        for i in range(num_layers):
            layer = WhisperDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=keras.activations.gelu,
                layer_norm_epsilon=1e-5,
                kernel_initializer=whisper_kernel_initializer(),
                normalize_first=True,
                dtype=dtype,
                name=f"transformer_decoder_layer_{i}",
            )
            self.decoder_transformer_layers.append(layer)
        self.decoder_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=dtype,
            name="decoder_layer_norm",
        )

        # === Functional Model ===
        # Note that the encoder does not have a padding mask:
        # https://github.com/openai/whisper/blob/v20230124/whisper/model.py#L132.
        encoder_feature_input = keras.Input(
            shape=(None, num_mels), dtype="float32", name="encoder_features"
        )
        decoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )
        # Encoder.
        # Embed the input features. This consists of two 1D convolutional
        # layers.
        # For the first layer, we use `padding="same"` since that corresponds to
        # a padding size of 1.
        embedded_features = keras.activations.gelu(
            self.encoder_conv_layer_1(encoder_feature_input),
            approximate=False,
        )
        # For the second conv. layer, we cannot use `padding="same"` since
        # that corresponds to a padding size of 1.5 (since stride is 2). Hence,
        # we will manually pad the input.
        embedded_features = self.encoder_padder(embedded_features)
        embedded_features = keras.activations.gelu(
            self.encoder_conv_layer_2(embedded_features),
            approximate=False,
        )
        # The position embedding layer for the encoder is a sinusoidal embedding
        # layer: https://github.com/openai/whisper/blob/v20230124/whisper/model.py#L137.
        # Hence, we set it to be non-trainable.
        # TODO: We can use `keras_hub.layers.SinePositionEncoding` layer.
        positions = self.encoder_position_embedding(embedded_features)
        x = self.encoder_embeddings_add((embedded_features, positions))
        x = self.encoder_embeddings_dropout(x)
        for transformer_layer in self.encoder_transformer_layers:
            x = transformer_layer(x)
        x = self.encoder_layer_norm(x)
        encoder_output = x
        # Decoder.
        x = self.decoder_embeddings(decoder_token_id_input)
        x = self.decoder_embeddings_dropout(x)
        for transformer_layer in self.decoder_transformer_layers:
            x = transformer_layer(
                decoder_sequence=x,
                encoder_sequence=encoder_output,
                decoder_padding_mask=decoder_padding_mask_input,
            )
        x = self.decoder_layer_norm(x)
        decoder_output = x
        super().__init__(
            inputs={
                "encoder_features": encoder_feature_input,
                "decoder_token_ids": decoder_token_id_input,
                "decoder_padding_mask": decoder_padding_mask_input,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
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
        self.num_mels = num_mels
        self.dropout = dropout
        self.max_encoder_sequence_length = max_encoder_sequence_length
        self.max_decoder_sequence_length = max_decoder_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_mels": self.num_mels,
                "dropout": self.dropout,
                "max_encoder_sequence_length": self.max_encoder_sequence_length,
                "max_decoder_sequence_length": self.max_decoder_sequence_length,
            }
        )
        return config
