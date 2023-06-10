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
import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.gpt_neox.rotary_embedding import RotaryEmbedding
from keras_nlp.utils.keras_utils import clone_initializer


class GPTNeoXAttention(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout=0.1,
        max_position_embeddings=512,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        **kwargs,
    ):

        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.rotary_pct = rotary_pct
        self.dropout = dropout
        self.attn_head_size = hidden_dim // num_heads
        self.rotary_dim = int(self.attn_head_size * rotary_pct)
        self.rotary_emb_base = rotary_emb_base
        self.max_position_embeddings = max_position_embeddings
        self.rotary_embedding = RotaryEmbedding(self.rotary_dim, rotary_emb_base)

        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)

        self._query_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="attention_output",
        )
        self._key_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="key",
        )
        self._value_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="value",
        )

        self._attn_dropout_layer = keras.layers.Dropout(
            self.dropout, name="attention_dropout"
        )

        self._softmax = keras.layers.Softmax(axis=-1, name="attention_softmax")

        # Output.
        self._output_dense = keras.layers.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, self.hidden_dim),
            bias_axes="d",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="attention_output",
        )

    def _get_common_kwargs_for_sublayer(self, use_bias=True):
        common_kwargs = {}

        kernel_initializer = clone_initializer(self._kernel_initializer)
        bias_initializer = clone_initializer(self._bias_initializer)

        common_kwargs["kernel_initializer"] = kernel_initializer
        if use_bias:
            common_kwargs["bias_initializer"] = bias_initializer

        return common_kwargs

    def _masked_softmax(self, attention_scores, attention_mask=None):
        #         print(attention_scores[0].shape, attention_scores[1].shape)
        # print(attention_mask.shape, attention_scores.shape)
        if attention_mask is not None:
            mask_expansion_axis = -3
            for _ in range(
                attention_scores.shape.rank - attention_mask.shape.rank
            ):
                attention_mask = tf.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):

        attention_scores = tf.einsum("aecd,abcd->acbe", key, query)

        #         batch_size, _, key_len, _ = tf.shape(key)
        #         causal_mask = compute_causal_mask(batch_size, key_len, key_len)
        #         attention_mask = attention_mask & causal_mask
        #         mask_value = tf.constant(float('-inf'), dtype=attention_scores.dtype)
        #         attention_scores = tf.where(causal_mask, attention_scores, mask_value)

        #         print(attention_scores[0].shape, attention_scores[1].shape)
        #         if attention_mask is not None:
        #             attention_scores += attention_mask

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._attn_dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum("acbe,aecd->abcd", attention_scores, value)

        return attention_output, attention_scores

    def call(self,
        hidden_states,
        attention_mask,
        return_attention_scores=False,
        training=None,
    ):

        query = self._query_dense(hidden_states)
        key = self._key_dense(hidden_states)
        value = self._value_dense(hidden_states)

        # query = tf.transpose(query, (0, 2, 1, 3))
        # key = tf.transpose(key, (0, 2, 1, 3))
        # value = tf.transpose(value, (0, 2, 1, 3))

        query_rot, query_pass = (
            query[..., : self.rotary_dim],
            query[..., self.rotary_dim :],
        )
        key_rot, key_pass = (
            key[..., : self.rotary_dim],
            key[..., self.rotary_dim :],
        )

        query, key = self.rotary_embedding(query_rot, key_rot)
        query = tf.concat((query, query_pass), axis=-1)
        key = tf.concat((key, key_pass), axis=-1)

        #         if layer_past is not None:
        #             past_key, past_value = layer_past
        #             key = tf.concat((past_key, key), axis=-2)
        #             value = tf.concat((past_value, value), axis=-2)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )

        # Reshape `attention_output` to `(batch_size, sequence_length, hidden_dim)`.
        attention_output = tf.reshape(
            attention_output,
            [
                tf.shape(attention_output)[0],
                tf.shape(attention_output)[1],
                self.hidden_dim,
            ],
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output