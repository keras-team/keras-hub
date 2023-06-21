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


"""XLNet Content and Query Embedding implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras


def xlnet_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


class ContentAndQueryEmbedding(keras.layers.Layer):
    """
    Content and Query Embedding.

    This class creates Content and Query Embeddings for XLNet model
    which is later used in XLNet ENcoder.

    In addition to that, it also creates relative_positional_encoding
    and processes attention masks for both states.

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
     - [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        dropout,
        kernel_initializer_range,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_initializer_range = kernel_initializer_range

        self.kernel_initializer = xlnet_kernel_initializer(
            self.kernel_initializer_range
        )
        self._built = None

    def positional_embedding(self, pos_seq, inv_freq, bsz=None):
        sinusoid_inp = tf.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = tf.concat(
            [tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], axis=-1
        )
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = tf.tile(pos_emb, [1, bsz, 1])

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None, clamp_len=-1):
        """create relative positional encoding."""
        freq_seq = tf.range(0, self.hidden_dim, 2.0)
        inv_freq = 1 / (10000 ** (freq_seq / self.hidden_dim))

        beg, end = klen, -qlen

        fwd_pos_seq = tf.range(beg, end, -1.0)
        if clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def build(self, input_shape):
        self.mask_emb = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="mask_emb",
        )
        self.word_embed = keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.hidden_dim,
            name="word_embedding",
        )
        self.dropout_layer = keras.layers.Dropout(self.dropout)

        super().build(input_shape)

    def call(
        self,
        token_id_input,
        padding_mask,
        token_type_id,
        mems,
        perm_mask,
        target_mapping,
    ):
        if not self._built:
            self.build((1, 1))
            self._built = True

        padding_mask = 1 - padding_mask
        padding_mask = tf.reshape(
            padding_mask, [tf.shape(padding_mask)[1], tf.shape(padding_mask)[0]]
        )
        perm_mask = tf.transpose(perm_mask, [1, 2, 0])
        target_mapping = tf.transpose(target_mapping, [1, 2, 0])

        bsz, qlen = tf.shape(token_id_input)[0], tf.shape(token_id_input)[1]

        mlen = (
            tf.shape(mems[0])[0]
            if mems is not None and mems[0] is not None
            else 0
        )
        klen = mlen + qlen

        if padding_mask is not None and perm_mask is not None:
            data_mask = padding_mask[None] + perm_mask
        elif padding_mask is not None and perm_mask is None:
            data_mask = padding_mask[None]
        elif padding_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            if mlen > 0:
                mems_mask = tf.zeros([tf.shape(data_mask)[0], mlen, bsz])
                data_mask = tf.concat(
                    [tf.cast(mems_mask, dtype=tf.int32), data_mask], axis=1
                )
            attn_mask = data_mask[:, :, :, None]
        else:
            attn_mask = None

        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=attn_mask.dtype)
            attn_mask_h = -tf.eye(qlen, dtype=attn_mask.dtype)
            if mlen > 0:
                attn_mask_h = tf.concat(
                    [
                        tf.zeros([qlen, mlen], dtype=attn_mask_h.dtype),
                        attn_mask_h,
                    ],
                    axis=-1,
                )

            attn_mask_h = tf.cast(
                (attn_mask + attn_mask_h[:, :, None, None]) > 0,
                dtype=attn_mask_h.dtype,
            )
        else:
            attn_mask_h = None

        # Word embeddings and prepare h & g hidden states
        word_emb = self.word_embed(token_id_input)
        output_h = self.dropout_layer(word_emb)
        if target_mapping is not None:
            word_emb_q = tf.tile(
                self.mask_emb, [tf.shape(target_mapping)[0], bsz, 1]
            )
            output_g = self.dropout_layer(word_emb_q)
        else:
            output_g = None

        token_type_id = (
            tf.transpose(token_type_id, perm=(1, 0))
            if token_type_id is not None
            else None
        )
        # Segment embedding
        if token_type_id is not None:
            if mlen > 0:
                mem_pad = tf.zeros([mlen, bsz], dtype=token_type_id.dtype)
                cat_ids = tf.concat([mem_pad, token_type_id], 0)
            else:
                cat_ids = token_type_id

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = tf.cast(
                tf.logical_not(
                    tf.equal(token_type_id[:, None], cat_ids[None, :])
                ),
                dtype=token_type_id.dtype,
            )
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout_layer(pos_emb)

        # to make sure inputs suitable for TwoStreamRelativeAttention
        output_g = (
            tf.reshape(
                output_g,
                [
                    tf.shape(output_g)[1],
                    tf.shape(output_g)[0],
                    tf.shape(output_g)[2],
                ],
            )
            if output_g is not None
            else None
        )
        attn_mask_h = (
            1.0
            - tf.cast(
                tf.transpose(tf.squeeze(attn_mask_h, -1), perm=[2, 0, 1]),
                tf.float32,
            )
            if attn_mask_h is not None
            else None
        )
        attn_mask = (
            1.0
            - tf.cast(
                tf.transpose(tf.squeeze(attn_mask, -1), perm=[2, 0, 1]),
                tf.float32,
            )
            if attn_mask is not None
            else None
        )
        pos_emb = tf.reshape(
            pos_emb,
            [tf.shape(pos_emb)[1], tf.shape(pos_emb)[0], tf.shape(pos_emb)[2]],
        )
        seg_mat = (
            tf.cast(tf.transpose(seg_mat, perm=[2, 0, 1]), dtype=tf.bool)
            if seg_mat is not None
            else None
        )
        target_mapping = (
            tf.cast(
                tf.reshape(
                    target_mapping,
                    [
                        tf.shape(target_mapping)[2],
                        tf.shape(target_mapping)[0],
                        tf.shape(target_mapping)[1],
                    ],
                ),
                tf.float32,
            )
            if target_mapping is not None
            else None
        )

        return (
            output_h,
            output_g,
            pos_emb,
            target_mapping,
            seg_mat,
            mems,
            attn_mask_h,
            attn_mask,
        )
