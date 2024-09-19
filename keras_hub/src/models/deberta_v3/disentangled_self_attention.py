# Copyright 2024 The KerasHub Authors
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

import keras
from keras import ops

from keras_hub.src.utils.keras_utils import clone_initializer


class DisentangledSelfAttention(keras.layers.Layer):
    """DisentangledSelfAttention layer.

    This is an implementation of disentangled self-attention as described in the
    paper ["DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing"](https://arxiv.org/abs/2111.09543).
    Effectively, this layer implements Multi-Head Self Attention with relative
    attention, i.e., to get the final attention score, we compute the
    content-to-position and position-to-content attention scores, and add these
    scores to the vanilla multi-head self-attention scores.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. Hidden dimension of the input, i.e., `hidden_states`.
        max_position_embeddings: int. The maximum input
            sequence length. Defaults to `512`.
        bucket_size: int. The size of the relative position
            buckets. Generally equal to `max_sequence_length // 2`.
            Defaults to `256`.
        dropout: float. Dropout probability. Defaults to `0.1`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense layers.
            Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense layers.
            Defaults to `"zeros"`.
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        max_position_embeddings=512,
        bucket_size=256,
        dropout=0.1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)

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

        # We have three types of attention - MHA, p2c and c2p.
        num_type_attn = 3
        self.scale_factor = 1.0 / math.sqrt(
            float(num_type_attn * self.attn_head_size)
        )

    def build(self, inputs_shape, rel_embeddings_shape=None):
        # Q, K, V linear layers.
        self._query_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)
        self._key_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)
        self._value_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        # Relative attention.
        self._position_dropout_layer = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
        )

        self._attn_dropout_layer = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
            name="attention_dropout",
        )
        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        # Output.
        self._output_dense = keras.layers.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, self.hidden_dim),
            bias_axes="d",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(inputs_shape)
        self.built = True

    def _get_common_kwargs_for_sublayer(self, use_bias=True):
        common_kwargs = {}

        kernel_initializer = clone_initializer(self._kernel_initializer)
        bias_initializer = clone_initializer(self._bias_initializer)

        common_kwargs["kernel_initializer"] = kernel_initializer
        if use_bias:
            common_kwargs["bias_initializer"] = bias_initializer

        return common_kwargs

    def _masked_softmax(self, attention_scores, attention_mask=None):
        """Normalizes the attention scores to probabilities using softmax.

        This implementation is the similar to the one present in
        `keras.layers.MultiHeadAttention`.
        """

        if attention_mask is not None:
            mask_expansion_axis = -3
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
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
        """Computes the attention score and returns the attended outputs.

        This function computes vanilla MHA score, and relative attention scores
        (p2c and c2p). It then sums them up to get the final attention score,
        which is used to compute the attended outputs.
        """

        attention_scores = ops.einsum(
            "aecd,abcd->acbe",
            key,
            query,
        )
        attention_scores = ops.multiply(attention_scores, self.scale_factor)

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

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._attn_dropout_layer(
            attention_scores, training=training
        )
        attention_output = ops.einsum(
            "acbe,aecd->abcd", attention_scores, value
        )

        return attention_output, attention_scores

    def _make_log_bucket_position(self, rel_pos):
        dtype = rel_pos.dtype
        sign = ops.sign(rel_pos)
        mid = self.bucket_size // 2
        mid = ops.cast(mid, dtype=dtype)

        # If `rel_pos[i][j]` is out of bounds, assign value `mid`.
        abs_pos = ops.where(
            condition=(rel_pos < mid) & (rel_pos > -mid),
            x1=mid - 1,
            x2=ops.abs(rel_pos),
        )

        def _get_log_pos(abs_pos, mid):
            numerator = ops.log(abs_pos / mid)
            numerator = numerator * ops.cast(mid - 1, dtype=numerator.dtype)
            denominator = ops.log((self.max_position_embeddings - 1) / mid)
            val = ops.ceil(numerator / denominator)
            val = ops.cast(val, dtype=mid.dtype)
            val = val + mid
            return val

        log_pos = _get_log_pos(abs_pos, mid)

        bucket_pos = ops.where(
            condition=abs_pos <= mid,
            x1=rel_pos,
            x2=log_pos * sign,
        )
        bucket_pos = ops.cast(bucket_pos, dtype="int")

        return bucket_pos

    def _get_rel_pos(self, num_positions):
        ids = ops.arange(num_positions)
        ids = ops.cast(ids, dtype="int")
        query_ids = ops.expand_dims(ids, axis=-1)
        key_ids = ops.expand_dims(ids, axis=0)
        key_ids = ops.repeat(key_ids, repeats=num_positions, axis=0)

        rel_pos = query_ids - key_ids
        rel_pos = self._make_log_bucket_position(rel_pos)

        rel_pos = ops.expand_dims(ops.expand_dims(rel_pos, axis=0), axis=0)
        return rel_pos

    def _compute_disentangled_attention(
        self,
        query,
        key,
        rel_embeddings,
    ):
        """Computes relative attention scores (p2c and c2p)."""

        batch_size = ops.shape(query)[0]
        num_positions = ops.shape(query)[1]

        rel_pos = self._get_rel_pos(num_positions)

        rel_attn_span = self.bucket_size
        score = 0

        pos_query = self._query_dense(rel_embeddings)
        pos_key = self._key_dense(rel_embeddings)

        # c2p
        c2p_attn_scores = ops.einsum(
            "aecd,abcd->acbe",
            pos_key,
            query,
        )
        c2p_pos = ops.clip(rel_pos + rel_attn_span, 0, rel_attn_span * 2 - 1)
        c2p_pos = ops.broadcast_to(
            c2p_pos,
            shape=(
                batch_size,
                self.num_heads,
                num_positions,
                num_positions,
            ),
        )

        if keras.config.backend() == "tensorflow":
            # Work around dynamic shape bug on tensorflow backend.
            import tensorflow as tf

            c2p_attn_scores = tf.gather(
                c2p_attn_scores,
                indices=c2p_pos,
                batch_dims=3,
            )
        else:
            c2p_attn_scores = ops.take_along_axis(
                c2p_attn_scores,
                indices=c2p_pos,
                axis=3,
            )
        c2p_attn_scores = ops.multiply(c2p_attn_scores, self.scale_factor)
        score += c2p_attn_scores

        # p2c
        p2c_attn_scores = ops.einsum(
            "aecd,abcd->acbe",
            pos_query,
            key,
        )
        p2c_pos = ops.clip(-rel_pos + rel_attn_span, 0, rel_attn_span * 2 - 1)
        p2c_pos = ops.broadcast_to(
            p2c_pos,
            shape=(
                batch_size,
                self.num_heads,
                num_positions,
                num_positions,
            ),
        )
        if keras.config.backend() == "tensorflow":
            # Work around dynamic shape bug on tensorflow backend.
            import tensorflow as tf

            p2c_attn_scores = tf.gather(
                p2c_attn_scores,
                indices=p2c_pos,
                batch_dims=3,
            )
        else:
            p2c_attn_scores = ops.take_along_axis(
                p2c_attn_scores,
                indices=p2c_pos,
                axis=3,
            )
        p2c_attn_scores = ops.transpose(p2c_attn_scores, [0, 1, 3, 2])
        p2c_attn_scores = ops.multiply(p2c_attn_scores, self.scale_factor)
        score += p2c_attn_scores

        return score

    def call(
        self,
        inputs,
        rel_embeddings,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
    ):
        # `query`, `key`, `value` shape:
        # `(batch_size, sequence_length, num_heads, attn_head_size)`.
        query = self._query_dense(inputs)
        key = self._key_dense(inputs)
        value = self._value_dense(inputs)

        attention_output, attention_scores = self._compute_attention(
            query=query,
            key=key,
            value=value,
            rel_embeddings=rel_embeddings,
            attention_mask=attention_mask,
            training=training,
        )

        # Reshape `attention_output` to `(batch_size, sequence_length, hidden_dim)`.
        attention_output = ops.reshape(
            attention_output,
            [
                ops.shape(attention_output)[0],
                ops.shape(attention_output)[1],
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
                "bucket_size": self.bucket_size,
                "dropout": self.dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self._kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self._bias_initializer
                ),
            }
        )
        return config
