# Copyright 2022 The KerasNLP Authors
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

"""BART backbone model."""

import copy

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.position_embedding import PositionEmbedding
from keras_nlp.layers.transformer_decoder import TransformerDecoder
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.bart.bart_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty


def bart_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras.utils.register_keras_serializable(package="keras_nlp")
class BartBackbone(Backbone):
    """BART encoder-decoder network.

    This class implements a Transformer-based encoder-decoder model as
    described in
    ["BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"](https://arxiv.org/abs/1910.13461).

    The default constructor gives a fully customizable, randomly initialized BART
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

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
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.

    Examples:
    ```python
    input_data = {
        "encoder_token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "encoder_padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "decoder_token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "decoder_padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], shape=(1, 12)
        ),
    }

    # Randomly initialized BART encoder-decoder model with a custom config
    model = keras_nlp.models.BartBackbone(
        vocabulary_size=50265,
        num_layers=6,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
    )
    output = model(input_data)
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
        max_sequence_length=1024,
        **kwargs,
    ):
        # Encoder inputs
        encoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        encoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )

        # Decoder inputs.
        decoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )

        # Token embedding layer. This layer is shared by encoder and decoder.
        token_embedding_layer = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=bart_kernel_initializer(),
            name="token_embedding",
        )

        # ===== Encoder =====

        # Embed tokens and positions.
        token_embedding = token_embedding_layer(encoder_token_id_input)
        # Position embedding parameters are not shared by encode and decoder.
        position_embedding = PositionEmbedding(
            initializer=bart_kernel_initializer(),
            sequence_length=max_sequence_length,
            name="encoder_position_embedding",
        )(token_embedding)

        # Sum, normalize and apply dropout to embeddings.
        x = keras.layers.Add()((token_embedding, position_embedding))
        x = keras.layers.LayerNormalization(
            name="encoder_embeddings_layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="encoder_embeddings_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                dropout=dropout,
                layer_norm_epsilon=1e-5,
                kernel_initializer=bart_kernel_initializer(),
                name=f"transformer_encoder_layer_{i}",
            )(x, padding_mask=encoder_padding_mask)

        encoder_output = x

        # ===== Decoder =====

        # Embed tokens and positions.
        token_embedding = token_embedding_layer(decoder_token_id_input)
        # Position embedding parameters are not shared by encode and decoder.
        position_embedding = PositionEmbedding(
            initializer=bart_kernel_initializer(),
            sequence_length=max_sequence_length,
            name="decoder_position_embedding",
        )(token_embedding)

        # Sum, normalize and apply dropout to embeddings.
        x = keras.layers.Add()((token_embedding, position_embedding))
        x = keras.layers.LayerNormalization(
            name="decoder_embeddings_layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="decoder_embeddings_dropout",
        )(x)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            transformer_decoder_layer = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                layer_norm_epsilon=1e-5,
                kernel_initializer=bart_kernel_initializer(),
                name=f"transformer_decoder_layer_{i}",
                has_cross_attention=True,
            )
            x = transformer_decoder_layer(
                decoder_sequence=x,
                encoder_sequence=encoder_output,
                decoder_padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
            )

        decoder_output = x

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "encoder_token_ids": encoder_token_id_input,
                "encoder_padding_mask": encoder_padding_mask,
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
        self.max_sequence_length = max_sequence_length

    def get_config(self):
        return {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
            "name": self.name,
            "trainable": self.trainable,
        }

    @property
    def token_embedding(self):
        return self.get_layer("token_embedding")

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
