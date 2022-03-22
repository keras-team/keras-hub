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
import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TransformerEncoder


class PositionalEmbedding(keras.layers.Layer):
    """The positional embedding class."""

    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TranslationModel(keras.Model):
    """The machine translation model.

    The model is an encoder-decoder structure model. The encoder is a stack of
    `keras_nlp.TransformerEncoder`, and the decoder is a stack of
    `keras_nlp.TransformerDecoder`. We also pass in the tokenizer for encoder
    and decoder so that during save/load, the tokenizer is also kept.
    """

    def __init__(
        self,
        encoder_tokenizer,
        decoder_tokenizer,
        num_encoders,
        num_decoders,
        num_heads,
        transformer_intermediate_dim,
        encoder_vocab_size,
        decoder_vocab_size,
        embed_dim,
        sequence_length,
    ):
        super().__init__()
        self.encoders = []
        self.decoders = []
        for _ in range(num_encoders):
            self.encoders.append(
                TransformerEncoder(
                    num_heads=num_heads,
                    intermediate_dim=transformer_intermediate_dim,
                )
            )
        for _ in range(num_decoders):
            self.decoders.append(
                TransformerDecoder(
                    num_heads=num_heads,
                    intermediate_dim=transformer_intermediate_dim,
                )
            )

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        self.encoder_embedding = PositionalEmbedding(
            sequence_length=sequence_length,
            vocab_size=encoder_vocab_size,
            embed_dim=embed_dim,
        )

        self.decoder_embedding = PositionalEmbedding(
            sequence_length=sequence_length,
            vocab_size=decoder_vocab_size,
            embed_dim=embed_dim,
        )

        self.dense = keras.layers.Dense(
            decoder_vocab_size,
            activation="softmax",
        )

    def call(self, inputs):
        encoder_input, decoder_input = (
            inputs["encoder_inputs"],
            inputs["decoder_inputs"],
        )
        encoded = self.encoder_embedding(encoder_input)
        for encoder in self.encoders:
            encoded = encoder(encoded)

        decoded = self.decoder_embedding(decoder_input)
        for decoder in self.decoders:
            decoded = decoder(
                decoded,
                encoded,
                use_causal_mask=True,
            )

        output = self.dense(decoded)
        return output
