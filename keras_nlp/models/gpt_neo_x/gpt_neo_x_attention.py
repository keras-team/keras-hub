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
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.gpt_neo_x.rotary_embedding import RotaryEmbedding
from keras_nlp.utils.keras_utils import clone_initializer


class GPTNeoXAttention(keras.layers.Layer):
    """GPTNeoXAttention layer.

    This is an implementation of disentangled attention as described in the
    paper ["GPT-NeoX-20B: An Open-Source Autoregressive Language Model"](https://arxiv.org/abs/2204.06745).
    Effectively, this layer implements Multi-Head Self Attention with rotary embedding,

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. Hidden dimension of the input, i.e., `hidden_states`.
        max_position_embeddings: int, defaults to 512. The maximum input
            sequence length.
        bucket_size: int, defaults to 256. The size of the relative position
            buckets. Generally equal to `max_sequence_length // 2`.
        dropout: float, defaults to 0.1. Dropout probability.
        kernel_initializer: string or `keras.initializers` initializer,
            defaults to "glorot_uniform". The kernel initializer for
            the dense layers.
        bias_initializer: string or `keras.initializers` initializer,
            defaults to "zeros". The bias initializer for the dense layers.
        rotary_max_wavelength: int. The maximum angular wavelength of the sine/cosine
            curves, for rotary embeddings. Defaults to 10000.
        rotary_percentage: float. The percentage by which query, key, value matrices are
            to be rotated
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout=0.1,
        max_sequence_length=512,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        rotary_percentage=0.25,
        rotary_max_wavelength=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.rotary_percentage = rotary_percentage
        self.dropout = dropout
        self.attn_head_size = hidden_dim // num_heads
        self.rotary_ndims = int(self.attn_head_size * rotary_percentage)
        self.rotary_max_wavelength = rotary_max_wavelength
        self.max_sequence_length = max_sequence_length
        self.rotary_embedding = RotaryEmbedding(
            self.rotary_ndims, rotary_max_wavelength
        )
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)

        self._qkv_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, 3 * self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            name="query",
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
        attention_scores /= tf.sqrt(self.attn_head_size)

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._attn_dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum("acbe,aecd->abcd", attention_scores, value)

        return attention_output, attention_scores

    def call(
        self,
        hidden_states,
        attention_mask,
        return_attention_scores=False,
        training=None,
    ):
        query_key_value = self._qkv_dense(hidden_states)

        query = query_key_value[..., : self.attn_head_size]
        key = query_key_value[
            ..., self.attn_head_size : 2 * self.attn_head_size
        ]
        value = query_key_value[..., 2 * self.attn_head_size :]

        query, key = self.rotary_embedding(query, key)

        attention_output, attention_scores = self._compute_attention(
            query=query,
            key=key,
            value=value,
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