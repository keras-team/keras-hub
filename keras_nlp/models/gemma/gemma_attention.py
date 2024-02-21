# Copyright 2024 The KerasNLP Authors
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

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.utils.keras_utils import clone_initializer


class CachedGemmaAttention(keras.layers.Layer):
    """A cached grouped query attention layer."""

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        kernel_initializer="glorot_uniform",
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.num_key_value_groups = num_query_heads // num_key_value_heads

    def build(self, inputs_shape):
        self.hidden_dim = inputs_shape[-1]

        self.query_dense = keras.layers.EinsumDense(
            "btd,ndh->btnh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
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
            equation="btnh,nhd->btd",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )
        self.softmax = keras.layers.Softmax(dtype="float32")
        self.built = True

    def _apply_rope(self, x, positions):
        """Rope rotate q or k."""
        # TODO: refactor to use RotaryEmbedding layer?
        max_wavelength = 10000
        x_shape = ops.shape(x)
        freq_exponents = (2.0 / x_shape[-1]) * ops.cast(
            ops.arange(x_shape[-1] // 2, dtype="float32"), self.compute_dtype
        )
        timescale = max_wavelength**freq_exponents
        radians = positions[..., None] / timescale[None, None, :]
        radians = radians[..., None, :]
        sin, cos = ops.sin(radians), ops.cos(radians)
        x1, x2 = ops.split(x, 2, axis=-1)
        # Avoid `ops.concatenate` for now, to avoid a obscure bug with XLA
        # compilation on jax. We should be able to remove this once the
        # following PR is in all jax releases we care about:
        # https://github.com/openxla/xla/pull/7875
        output = ops.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        return ops.reshape(output, x_shape)

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
    ):
        query_normalization = 1 / np.sqrt(self.head_dim)

        q *= ops.cast(query_normalization, dtype=q.dtype)
        q_shape = ops.shape(q)
        q = ops.reshape(
            q,
            (
                *q_shape[:-2],
                self.num_key_value_heads,
                self.num_query_heads // self.num_key_value_heads,
                q_shape[-1],
            ),
        )
        b, q_len, _, _, h = ops.shape(q)

        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)
        attention_mask = attention_mask[:, None, None, :, :]
        orig_dtype = attention_logits.dtype
        attention_softmax = self.softmax(attention_logits, mask=attention_mask)
        attention_softmax = ops.cast(attention_softmax, orig_dtype)

        if self.dropout:
            attention_softmax = self.dropout_layer(
                attention_softmax, training=training
            )

        results = ops.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return ops.reshape(results, (b, q_len, self.num_query_heads, h))

    def call(
        self,
        x,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        training=False,
    ):
        seq_len = ops.shape(x)[1]
        start_index = cache_update_index
        positions = ops.cast(
            ops.arange(seq_len, dtype="float32"), self.compute_dtype
        )
        positions = positions + ops.cast(start_index, self.compute_dtype)
        query = self.query_dense(x)
        query = self._apply_rope(query, positions)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            key_update = self.key_dense(x)
            key_update = self._apply_rope(key_update, positions)
            value_update = self.value_dense(x)
            start = [0, cache_update_index, 0, 0]
            key = ops.slice_update(key_cache, start, key_update)
            value = ops.slice_update(value_cache, start, value_update)
            cache = ops.stack((key, value), axis=1)
        else:
            key = self.key_dense(x)
            key = self._apply_rope(key, positions)
            value = self.value_dense(x)

        attention_vec = self._compute_attention(
            query, key, value, attention_mask, training=training
        )

        # Wipe attn vec if there are no attended tokens.
        no_attended_tokens = ops.all(
            ops.equal(attention_mask, 0), axis=-1, keepdims=True
        )[..., None]
        attention_vec = ops.where(
            no_attended_tokens, ops.zeros_like(attention_vec), attention_vec
        )

        attention_output = self.output_dense(attention_vec)

        if cache is not None:
            return attention_output, cache
        return attention_output
