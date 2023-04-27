# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whisper backbone model."""


import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.position_embedding import PositionEmbedding
from keras_nlp.layers.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.whisper.whisper_decoder import WhisperDecoder
from keras_nlp.models.whisper.whisper_encoder import WhisperEncoder

# We hardcode the number of mel-frequency filters:
# https://github.com/openai/whisper/blob/v20230124/whisper/audio.py#L101-L102.
# TODO: If needed, we can make it configurable.
NUM_MELS = 80


def whisper_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras.utils.register_keras_serializable(package="keras_nlp")
class WhisperBackbone(Backbone):
    """A Whisper encoder-decoder network for speech.

    This class implements a Transformer-based encoder-decoder model as
    described in
    ["Robust Speech Recognition via Large-Scale Weak Supervision"](https://arxiv.org/abs/2212.04356).
    It includes the embedding lookups and transformer layers, but not the head
    for predicting the next token.

    The default constructor gives a fully customizable, randomly initialized Whisper
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset()` constructor.

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
        dropout: float. Dropout probability for the Transformer encoder.
        max_encoder_sequence_length: int. The maximum sequence length that the
            audio encoder can consume. Since the second convolutional layer in
            the encoder reduces the sequence length by half (stride of 2), we
            use `max_encoder_sequence_length // 2` as the sequence length for the
            positional embedding layer.
        max_decoder_sequence_length: int. The maximum sequence length that the
            text decoder can consume.

    Examples:
    ```python
    input_data = {
        "encoder_features": tf.ones(shape=(1, 12, 80), dtype=tf.int64),
        "decoder_token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "decoder_padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], shape=(1, 12)
        ),
    }

    # Randomly initialized Whisper encoder-decoder model with a custom config.
    model = keras_nlp.models.WhisperBackbone(
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
        dropout=0.0,
        max_encoder_sequence_length=3000,
        max_decoder_sequence_length=448,
        **kwargs,
    ):
        # Encoder inputs. Note that the encoder does not have a padding mask:
        # https://github.com/openai/whisper/blob/v20230124/whisper/model.py#L132.
        encoder_feature_input = keras.Input(
            shape=(None, NUM_MELS), dtype="float32", name="encoder_features"
        )

        # Decoder inputs.
        decoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )

        # ====== Encoder ======

        # Embed the input features. This consists of two 1D convolutional
        # layers.
        # For the first layer, we use `padding="same"` since that corresponds to
        # a padding size of 1.
        encoder_conv_layer_1 = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            name="encoder_token_embedding_conv_layer_1",
        )
        embedded_features = keras.activations.gelu(
            encoder_conv_layer_1(encoder_feature_input),
            approximate=False,
        )

        # For the second conv. layer, we cannot use `padding="same"` since
        # that corresponds to a padding size of 1.5 (since stride is 2). Hence,
        # we will manually pad the input.
        embedded_features = tf.pad(
            embedded_features, paddings=[[0, 0], [1, 1], [0, 0]]
        )
        encoder_conv_layer_2 = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=3,
            strides=2,
            padding="valid",
            name="encoder_token_embedding_conv_layer_2",
        )
        embedded_features = keras.activations.gelu(
            encoder_conv_layer_2(embedded_features),
            approximate=False,
        )

        # The position embedding layer for the encoder is a sinusoidal embedding
        # layer: https://github.com/openai/whisper/blob/v20230124/whisper/model.py#L137.
        # Hence, we set it to be non-trainable.
        # TODO: We can use `keras_nlp.layers.SinePositionEncoding` layer.
        position_embedding = PositionEmbedding(
            initializer=whisper_kernel_initializer(),
            sequence_length=max_encoder_sequence_length // 2,
            name="encoder_position_embedding",
            trainable=False,
        )(embedded_features)

        # Sum and apply dropout to embeddings.
        x = keras.layers.Add()((embedded_features, position_embedding))
        x = keras.layers.Dropout(
            dropout,
            name="encoder_embeddings_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = WhisperEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                layer_norm_epsilon=1e-5,
                dropout=dropout,
                kernel_initializer=whisper_kernel_initializer(),
                normalize_first=True,
                name=f"transformer_encoder_layer_{i}",
            )(x)

        x = keras.layers.LayerNormalization(
            name="encoder_layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype=tf.float32,
        )(x)
        encoder_output = x

        # ====== Decoder ======

        # Embed tokens and positions.
        x = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_decoder_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=whisper_kernel_initializer(),
            name="decoder_token_and_position_embedding",
        )(decoder_token_id_input)

        # Apply dropout to embeddings.
        x = keras.layers.Dropout(
            dropout,
            name="decoder_embeddings_dropout",
        )(x)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            transformer_decoder_layer = WhisperDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                layer_norm_epsilon=1e-5,
                kernel_initializer=whisper_kernel_initializer(),
                normalize_first=True,
                name=f"transformer_decoder_layer_{i}",
                has_cross_attention=True,
            )
            x = transformer_decoder_layer(
                decoder_sequence=x,
                encoder_sequence=encoder_output,
                decoder_padding_mask=decoder_padding_mask,
            )

        x = keras.layers.LayerNormalization(
            name="decoder_layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype=tf.float32,
        )(x)
        decoder_output = x

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "encoder_features": encoder_feature_input,
                "decoder_token_ids": decoder_token_id_input,
                "decoder_padding_mask": decoder_padding_mask,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
            **kwargs,
        )

        # All references to `self` below this line
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
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
                "dropout": self.dropout,
                "max_encoder_sequence_length": self.max_encoder_sequence_length,
                "max_decoder_sequence_length": self.max_decoder_sequence_length,
            }
        )
        return config

    @property
    def token_embedding(self):
        return self.get_layer(
            "decoder_token_and_position_embedding"
        ).token_embedding
