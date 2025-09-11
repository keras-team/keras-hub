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

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer


class GptOssAttention(keras.layers.Layer):
    """A cached attention layer with sliding window and sink tokens.

    This layer implements the attention mechanism described in the GPT-OSS
    paper. It includes grouped-query attention, rotary position embeddings,
    sliding window attention, and sink tokens for improved performance on
    long sequences.

    Args:
        num_query_heads (int): The number of query attention heads.
        num_key_value_heads (int): The number of key and value attention
            heads.
        rope_max_wavelength (int, optional): The maximum wavelength for the
            rotary position embedding. Defaults to 10000.
        rope_scaling_factor (float, optional): The scaling factor for the
            rotary position embedding. Defaults to 1.0.
        kernel_initializer (str, optional): The initializer for the kernel
            weights. Defaults to "glorot_uniform".
        sliding_window (int, optional): The size of the sliding window.
            Defaults to 4096.
        dropout (float, optional): The dropout rate. Defaults to 0.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        kernel_initializer="glorot_uniform",
        sliding_window=4096,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.dropout = dropout

        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

        self.rope_scaling_factor = rope_scaling_factor

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        self._hidden_dim = inputs_shape[-1]
        self._head_dim = self._hidden_dim // self.num_query_heads
        self._inv_norm_factor = 1.0 / math.sqrt(self._head_dim)

        # Calculate rotary dimension - use the largest even number <= head_dim
        self._rotary_dim = (self._head_dim // 2) * 2

        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self._head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                self._head_dim,
            ),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                self._head_dim,
            ),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self.output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, self._hidden_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (None, None, self.num_query_heads, self._head_dim)
        )

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            dtype=self.dtype_policy,
        )

        self.sinks = self.add_weight(
            shape=(self.num_query_heads,),
            initializer="random_normal",
            dtype=self.dtype,
            name="sinks",
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self.query_dense(hidden_states)

        # Compute RoPE for queries (only apply to first _rotary_dim dimensions)
        if self._rotary_dim < self._head_dim:
            query_rot = query[..., : self._rotary_dim]
            query_rot = self.rotary_embedding_layer(
                query_rot, start_index=start_index
            )
            query = ops.concatenate(
                [query_rot, query[..., self._rotary_dim :]], axis=-1
            )
        else:
            query = self.rotary_embedding_layer(query, start_index=start_index)

        def _compute_key_value(x):
            key, value = self.key_dense(x), self.value_dense(x)
            # Compute RoPE for keys (only apply to first _rotary_dim dimensions)
            if self._rotary_dim < self._head_dim:
                key_rot = key[..., : self._rotary_dim]
                key_rot = self.rotary_embedding_layer(
                    key_rot, start_index=start_index
                )
                key = ops.concatenate(
                    [key_rot, key[..., self._rotary_dim :]], axis=-1
                )
            else:
                key = self.rotary_embedding_layer(key, start_index=start_index)
            return key, value

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update, value_update = _compute_key_value(hidden_states)
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key, value = _compute_key_value(hidden_states)

        # [batch_shape, seq_len, num_key_value_heads, head_dim]
        # -> [batch_shape, seq_len, num_heads, head_dim]
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query, key, value, attention_mask
        )

        attention_output = self.dropout_layer(
            attention_output, training=training
        )

        attention_output = self.output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _compute_attention(self, query, key, value, attention_mask=None):
        attention_scores = ops.einsum(self._dot_product_equation, query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )

        if attention_mask is not None:
            # The mask is a boolean tensor, True for positions to be masked.
            # We add a large negative number to the masked positions.
            # Use a large negative value for masking
            if self.compute_dtype == "float32":
                adder = ops.cast(-1e9, self.compute_dtype)
            else:
                adder = ops.cast(-1e4, self.compute_dtype)
            attention_scores = ops.where(
                attention_mask[:, None, :, :], adder, attention_scores
            )

        # Handle sink tokens by concatenating them to the logits.
        b = ops.shape(query)[0]
        q = ops.shape(query)[1]
        sinks = ops.reshape(self.sinks, (1, self.num_query_heads, 1, 1))
        sinks = ops.broadcast_to(sinks, (b, self.num_query_heads, q, 1))
        # attention_scores shape: [b, num_heads, q, k]
        # sinks shape: [b, num_heads, q, 1]
        # We need to concatenate along the last dimension
        combined_logits = ops.concatenate([attention_scores, sinks], axis=-1)

        # Stabilize logits before softmax for numerical stability.
        max_logits = ops.max(combined_logits, axis=-1, keepdims=True)
        max_logits = ops.stop_gradient(max_logits)
        combined_logits = combined_logits - max_logits

        probs = ops.softmax(combined_logits, axis=-1)

        # Remove the sink probabilities before computing the output.
        attention_scores = probs[..., :-1]
        attention_scores = ops.cast(attention_scores, self.compute_dtype)

        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )

        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "kernel_initializer": keras.initializers.serialize(
                    self._kernel_initializer
                ),
                "sliding_window": self.sliding_window,
                "dropout": self.dropout,
            }
        )
        return config
