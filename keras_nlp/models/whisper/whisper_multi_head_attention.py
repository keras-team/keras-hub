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
"""Whisper Multi-Head Attention layer."""


import string

import tensorflow as tf
from keras.layers import MultiHeadAttention
from keras.layers import core
from keras.utils import tf_utils

_CHR_IDX = string.ascii_lowercase


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """Builds an einsum equation for projections inside multi-head attention."""
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = f"{input_str},{kernel_str}->{output_str}"

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


class WhisperMultiHeadAttention(MultiHeadAttention):
    """Whisper Multi-Head Attention layer.

    Inherits from `keras.layers.MultiHeadAttention`, and overrides the
    `_build_from_signature` method so that Q, V projection layers have bias
    whereas K projection layer does not.
    """

    def _build_from_signature(self, query, value, key=None):
        """Builds layers and variables.

        Once the method is called, self._built_from_signature will be set to
        True.

        Args:
          query: Query tensor or TensorShape.
          value: Value tensor or TensorShape.
          key: Key tensor or TensorShape.
        """
        self._built_from_signature = True
        if hasattr(query, "shape"):
            self._query_shape = tf.TensorShape(query.shape)
        else:
            self._query_shape = tf.TensorShape(query)
        if hasattr(value, "shape"):
            self._value_shape = tf.TensorShape(value.shape)
        else:
            self._value_shape = tf.TensorShape(value)
        if key is None:
            self._key_shape = self._value_shape
        elif hasattr(key, "shape"):
            self._key_shape = tf.TensorShape(key.shape)
        else:
            self._key_shape = tf.TensorShape(key)

        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf_utils.maybe_init_scope(self):
            free_dims = self._query_shape.rank - 1
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                free_dims, bound_dims=1, output_dims=2
            )
            self._query_dense = core.EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1, [self._num_heads, self._key_dim]
                ),
                bias_axes=bias_axes if self._use_bias else None,
                name="query",
                **self._get_common_kwargs_for_sublayer(),
            )
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._key_shape.rank - 1, bound_dims=1, output_dims=2
            )
            self._key_dense = core.EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1, [self._num_heads, self._key_dim]
                ),
                name="key",
                **self._get_common_kwargs_for_sublayer(),
            )
            einsum_equation, bias_axes, output_rank = _build_proj_equation(
                self._value_shape.rank - 1, bound_dims=1, output_dims=2
            )
            self._value_dense = core.EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1, [self._num_heads, self._value_dim]
                ),
                bias_axes=bias_axes if self._use_bias else None,
                name="value",
                **self._get_common_kwargs_for_sublayer(),
            )

            # Builds the attention computations for multi-head dot product
            # attention.  These computations could be wrapped into the keras
            # attention layer once it supports mult-head einsum computations.
            self._build_attention(output_rank)
            self._output_dense = self._make_output_dense(
                free_dims,
                self._get_common_kwargs_for_sublayer(),
                "attention_output",
            )
