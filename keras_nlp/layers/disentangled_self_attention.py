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
import math

import tensorflow as tf
from tensorflow import keras

from keras_nlp.utils.keras_utils import clone_initializer


def torch_gather(x, indices, gather_axis):
    if gather_axis < 0:
        gather_axis = tf.rank(x) + gather_axis
    if gather_axis != tf.rank(x) - 1:
        pre_roll = tf.rank(x) - 1 - gather_axis
        permutation = tf.roll(tf.range(tf.rank(x)), pre_roll, axis=0)
        x = tf.transpose(x, perm=permutation)
        indices = tf.transpose(indices, perm=permutation)
    else:
        pre_roll = 0
    flat_x = tf.reshape(x, (-1, tf.shape(x)[-1]))
    flat_indices = tf.reshape(indices, (-1, tf.shape(indices)[-1]))
    gathered = tf.gather(flat_x, flat_indices, batch_dims=1)
    gathered = tf.reshape(gathered, tf.shape(indices))

    if pre_roll != 0:
        permutation = tf.roll(tf.range(tf.rank(x)), -pre_roll, axis=0)
        gathered = tf.transpose(gathered, perm=permutation)
    return gathered


class DisentangledSelfAttention(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        max_position_embeddings=512,
        dropout=0.1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # Passed args.
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout

        # Initializers.
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)

        # Derived args.
        self.max_relative_positions = max_position_embeddings

        self.attn_head_size = hidden_dim // num_heads

        self.scale_factor = 1.0 / math.sqrt(float(3 * self.attn_head_size))

        # Layers.

        # Q, K, V linear layers.
        self._query_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="query",
        )
        self._key_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes=None,
            **self._get_common_kwargs_for_sublayer(use_bias=False),
            name="key",
        )
        self._value_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="value",
        )

        # Relative attention.
        self._position_dropout_layer = keras.layers.Dropout(self.dropout)
        # For context->position.
        self._c2p_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes=None,
            **self._get_common_kwargs_for_sublayer(use_bias=False),
            name="c2p",
        )
        # For position->context.
        self._p2c_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="p2c",
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
        # Normalize the attention scores to probabilities.
        # `attention_scores` = [B, N, T, S]
        if attention_mask is not None:
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = tf.expand_dims(attention_mask, axis=-3)
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(
        self,
        query,
        key,
        value,
        rel_embeddings,
        rel_pos=None,
        attention_mask=None,
        training=None,
    ):
        query = tf.multiply(query, self.scale_factor)
        # `attention_scores` -> `(batch_size, num_heads, sequence_length, sequence_length)`
        attention_scores = tf.einsum(
            "aecd,abcd->acbe",
            key,
            query,
        )

        rel_embeddings = self._position_dropout_layer(
            rel_embeddings,
            training=training,
        )
        rel_attn_scores = self._compute_disentangled_attention(
            query=query,
            key=key,
            rel_embeddings=rel_embeddings,
            rel_pos=rel_pos,
        )
        if rel_attn_scores is not None:
            attention_scores += rel_attn_scores
        attention_scores = self._masked_softmax(attention_scores)
        attention_scores = self._attn_dropout_layer(attention_scores)

        attention_output = tf.einsum("acbe,aecd->abcd", attention_scores, value)

        return attention_output, attention_scores

    def _get_rel_pos_ids(self, query_length, key_length):
        query_ids = tf.range(query_length, dtype=tf.int64)[:, tf.newaxis]
        key_ids = tf.range(key_length, dtype=tf.int64)[tf.newaxis, :]
        rel_pos_ids = query_ids - tf.tile(key_ids, [query_length, 1])
        rel_pos_ids = rel_pos_ids[:query_length, :]
        return rel_pos_ids

    def _get_rel_attn_span(self, query_length, key_length):
        rel_attn_span = tf.maximum(query_length, key_length)
        rel_attn_span = tf.minimum(rel_attn_span, self.max_relative_positions)
        rel_attn_span = tf.cast(rel_attn_span, dtype=tf.int64)
        return rel_attn_span

    def _compute_disentangled_attention(
        self,
        query,
        key,
        rel_embeddings,
        rel_pos=None,
    ):

        batch_size = tf.shape(query)[0]
        query_length = tf.shape(query)[-3]
        key_length = tf.shape(key)[-3]
        if rel_pos is None:
            rel_pos = self._get_rel_pos_ids(query_length, key_length)
        if rel_pos.shape.rank == 2:
            rel_pos = rel_pos[tf.newaxis, tf.newaxis, :, :]
        elif rel_pos.shape.rank == 3:
            rel_pos = rel_pos[tf.newaxis, :, :, :]
        elif rel_pos.shape.rank != 4:
            raise ValueError("`rel_pos` must be of rank 2 or 3 or 4.")

        rel_attn_span = self._get_rel_attn_span(query_length, key_length)
        rel_embeddings = rel_embeddings[
            self.max_relative_positions
            - rel_attn_span : self.max_relative_positions
            + rel_attn_span,
            :,
        ]
        rel_embeddings = rel_embeddings[tf.newaxis, :]

        score = 0

        # c2p
        # `pos_key` is of shape `(1, 2 * rel_attn_span, num_heads, attn_head_size)`.
        pos_key = self._c2p_dense(rel_embeddings)
        c2p_attn_scores = tf.einsum(
            "abcd,efcd->acbf",
            query,
            pos_key,
        )
        c2p_pos = tf.clip_by_value(
            rel_pos + rel_attn_span, 0, rel_attn_span * 2 - 1
        )
        c2p_attn_scores = torch_gather(
            c2p_attn_scores,
            indices=tf.broadcast_to(
                c2p_pos,
                shape=(batch_size, self.num_heads, query_length, query_length),
            ),
            gather_axis=-1,
        )
        score += c2p_attn_scores

        # p2c
        pos_query = self._p2c_dense(rel_embeddings)
        pos_query = tf.multiply(pos_query, self.scale_factor)
        p2c_attn_scores = tf.einsum(
            "abcd,efcd->acbf",
            key,
            pos_query,
        )
        p2c_pos = tf.clip_by_value(
            -rel_pos + rel_attn_span, 0, rel_attn_span * 2 - 1
        )
        p2c_attn_scores = torch_gather(
            p2c_attn_scores,
            indices=tf.broadcast_to(
                p2c_pos,
                shape=(batch_size, self.num_heads, key_length, key_length),
            ),
            gather_axis=-1,
        )
        score += p2c_attn_scores

        return score

    def call(
        self,
        hidden_states,
        rel_embeddings,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
    ):
        # `query`, `key`, `value` are of shape
        # `(batch_size, sequence_length, num_heads, attn_head_size)`.
        query = self._query_dense(hidden_states)
        key = self._key_dense(hidden_states)
        value = self._value_dense(hidden_states)

        attention_output, attention_scores = self._compute_attention(
            query=query,
            key=key,
            value=value,
            rel_embeddings=rel_embeddings,
            rel_pos=None,
            attention_mask=attention_mask,
            training=training,
        )

        # Reshape `attention_output` to `(batch_size, sequence_length, hidden_dim)`.
        attention_output = tf.reshape(
            attention_output, attention_output.shape.as_list()[:-2] + [-1]
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "dropout": self.dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config
