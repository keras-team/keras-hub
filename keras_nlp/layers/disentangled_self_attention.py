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
        bucket_size=256,
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
        self.bucket_size = bucket_size
        self.dropout = dropout

        # Initializers.
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)

        # Derived args.
        self.attn_head_size = hidden_dim // num_heads

        self.scale_factor = 1.0 / math.sqrt(float(self.attn_head_size))

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

        # Relative attention.
        self._position_dropout_layer = keras.layers.Dropout(self.dropout)

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
        )
        if rel_attn_scores is not None:
            attention_scores += rel_attn_scores

        attention_scores = self._masked_softmax(attention_scores)
        attention_scores = self._attn_dropout_layer(
            attention_scores, training=training
        )

        attention_output = tf.einsum("acbe,aecd->abcd", attention_scores, value)

        return attention_output, attention_scores

    def _make_log_bucket_position(self, rel_pos):
        dtype = rel_pos.dtype
        sign = tf.math.sign(rel_pos)
        mid = self.bucket_size // 2
        mid = tf.cast(mid, dtype=dtype)

        # If `rel_pos[i][j]` is out of bounds, assign value `mid`.
        abs_pos = tf.where(
            condition=(rel_pos < mid) & (rel_pos > -mid),
            x=mid - 1,
            y=tf.math.abs(rel_pos),
        )

        def _get_log_pos(abs_pos, mid):
            numerator = tf.math.log(abs_pos / mid)
            numerator = numerator * tf.cast(mid - 1, dtype=numerator.dtype)
            denominator = tf.math.log((self.max_position_embeddings - 1) / mid)
            val = tf.math.ceil(numerator / denominator)
            val = tf.cast(val, dtype=mid.dtype)
            val = val + mid
            return val

        log_pos = _get_log_pos(abs_pos, mid)

        bucket_pos = tf.where(
            condition=abs_pos <= mid,
            x=rel_pos,
            y=log_pos * sign,
        )
        bucket_pos = tf.cast(bucket_pos, dtype=tf.int64)

        return bucket_pos

    def _get_rel_pos(self, num_positions):
        ids = tf.range(num_positions, dtype=tf.int64)
        query_ids = ids[:, tf.newaxis]
        # query_ids = tf.repeat(query_ids, repeats=num_positions, axis=1)
        key_ids = ids[tf.newaxis, :]
        key_ids = tf.repeat(key_ids, repeats=num_positions, axis=0)

        rel_pos = query_ids - key_ids
        rel_pos = self._make_log_bucket_position(rel_pos)

        rel_pos = rel_pos[tf.newaxis, tf.newaxis, :, :]
        return rel_pos

    def _compute_disentangled_attention(
        self,
        query,
        key,
        rel_embeddings,
    ):

        batch_size = tf.shape(query)[0]
        num_positions = tf.shape(query)[1]

        rel_pos = self._get_rel_pos(num_positions)

        rel_attn_span = self.bucket_size
        rel_embeddings = rel_embeddings[
            self.bucket_size + rel_attn_span : self.bucket_size + rel_attn_span,
            :,
        ]
        rel_embeddings = rel_embeddings[tf.newaxis, :]

        score = 0

        # def _reshape_rel_dense_output(inputs):
        #     # `inputs` is of shape `(batch_size, seq_length, num_heads, attn_head_size)`.
        #     # Reshape this to `(batch_size * num_heads, seq_length, attn_head_size)`.
        #     inputs = tf.transpose(inputs, perm=(0, 2, 1, 3))
        #     inputs = tf.reshape(inputs, (-1, tf.shape(inputs)[-2], tf.shape(inputs)[-1]))
        #     return inputs

        pos_query = self._query_dense(rel_embeddings)
        pos_query = tf.repeat(pos_query, repeats=batch_size, axis=0)

        pos_key = self._key_dense(rel_embeddings)
        pos_key = tf.repeat(pos_key, repeats=batch_size, axis=0)

        # c2p
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
                shape=(
                    batch_size,
                    self.num_heads,
                    num_positions,
                    num_positions,
                ),
            ),
            gather_axis=-1,
        )
        score += c2p_attn_scores / self.scale_factor

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
                shape=(
                    batch_size,
                    self.num_heads,
                    num_positions,
                    num_positions,
                ),
            ),
            gather_axis=-1,
        )
        score += p2c_attn_scores / self.scale_factor

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
            attention_mask=attention_mask,
            training=training,
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
