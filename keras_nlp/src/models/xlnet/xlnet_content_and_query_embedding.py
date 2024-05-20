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

import keras
from keras import ops


class ContentAndQueryEmbedding(keras.layers.Layer):
    """
    Content and Query Embedding.

    This class creates Content and Query Embeddings for XLNet model
    which is later used in XLNet Encoder.

    Args:
        vocabulary_size: int, number of tokens in the vocabulary.
        hidden_dim: int, the size hidden states.
        dropout: float, defaults to 0. the dropout value, shared by
            `keras.layers.TwoStreamRelativeAttention` and feedforward network.
        kernel_initializer_range: int, defaults to 0.02. The kernel initializer
            range for the dense and relative attention layers.
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    References:
     - [XLNet: Generalized Autoregressive Pretraining for Language Understanding]
     (https://arxiv.org/abs/1906.08237)
    """

    def __init__(
        self, vocabulary_size, hidden_dim, dropout, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def positional_embedding(self, pos_seq, inv_freq, bsz=None):
        sinusoid_inp = ops.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = ops.concatenate(
            [ops.sin(sinusoid_inp), ops.cos(sinusoid_inp)], axis=-1
        )
        pos_emb = ops.expand_dims(pos_emb, 1)
        pos_emb = (
            ops.ones(
                [
                    ops.shape(pos_emb)[0],
                    ops.shape(pos_emb)[1] * bsz,
                    ops.shape(pos_emb)[2],
                ],
                dtype=self.compute_dtype,
            )
            * pos_emb
        )

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None, clamp_len=-1):
        """create relative positional encoding."""
        freq_seq = ops.arange(0, self.hidden_dim, 2.0, dtype="float32")
        freq_seq = ops.cast(freq_seq, self.compute_dtype)
        inv_freq = 1 / (10000 ** (freq_seq / self.hidden_dim))

        beg, end = klen, -qlen

        fwd_pos_seq = ops.arange(beg, end, -1.0, dtype="float32")
        fwd_pos_seq = ops.cast(fwd_pos_seq, self.compute_dtype)
        if clamp_len > 0:
            fwd_pos_seq = ops.clip(
                fwd_pos_seq, x_min=-clamp_len, x_max=clamp_len
            )
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def build(self, input_shape):
        self.word_embed = keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="word_embedding",
        )
        self.word_embed.build(input_shape)
        self.dropout_layer = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
        )
        super().build(input_shape)

    def call(
        self,
        token_id_input,
        mlen=None,
    ):
        mlen = 0 if mlen is None else mlen

        bsz, qlen = ops.shape(token_id_input)[0], ops.shape(token_id_input)[1]
        klen = mlen + qlen

        # Word embeddings and prepare h & g hidden states
        word_emb = self.word_embed(token_id_input)
        word_emb = self.dropout_layer(word_emb)

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout_layer(pos_emb)
        pos_emb = ops.reshape(
            pos_emb,
            [
                ops.shape(pos_emb)[1],
                ops.shape(pos_emb)[0],
                ops.shape(pos_emb)[2],
            ],
        )

        return word_emb, pos_emb

    def compute_output_shape(self, token_id_input_shape):
        return [
            token_id_input_shape + (self.hidden_dim,),
            (token_id_input_shape[0], 1, self.hidden_dim),
        ]
