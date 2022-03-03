from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TransformerEncoder

import tensorflow as tf
from tensorflow import keras


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
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
    def __init__(
        self,
        num_encoders,
        num_decoders,
        encoder_tokenizer,
        decoder_tokenizer,
        embedding_params,
        encoder_params,
        decoder_params,
    ):
        super(TranslationModel, self).__init__()
        self.encoders = []
        self.decoders = []
        for _ in range(num_encoders):
            self.encoders.append(TransformerEncoder(**encoder_params))
        for _ in range(num_decoders):
            self.decoders.append(TransformerDecoder(**decoder_params))
            
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        self.encoder_embedding = PositionalEmbedding(
            embedding_params["sequence_length"],
            embedding_params["vocab_size"],
            embedding_params["embed_dim"],
        )

        self.decoder_embedding = PositionalEmbedding(
            embedding_params["sequence_length"],
            embedding_params["vocab_size"],
            embedding_params["embed_dim"],
        )

        self.dense = keras.layers.Dense(
            embedding_params["vocab_size"], activation="softmax")

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
    