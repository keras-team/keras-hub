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
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_nlp.utils.keras_utils import clone_initializer


class LlamaAttention(keras.layers.Layer):
    """Grouped query attention for Llama models"""

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_scaling_factor=1.0,
        kernel_initializer="glorot_uniform",
        rope_max_wavelength=10000,
        max_sequence_length=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_key_value_groups = num_query_heads // num_key_value_heads

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.max_sequence_length = max_sequence_length

        self.rope_scaling_factor = rope_scaling_factor
        self.rope_max_wavelength = rope_max_wavelength

    def build(self, inputs_shape):
        self.hidden_dim = inputs_shape[-1]
        self.attn_head_size = self.hidden_dim // self.num_query_heads

        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.attn_head_size),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)
        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.attn_head_size),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.attn_head_size),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        self._output_dense = keras.layers.EinsumDense(
            equation="bqm,mh->bqh",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(inputs_shape)

        self._rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            dtype=self.dtype_policy,
        )
        self._rotary_embedding_layer.build(inputs_shape)

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        query = self._query_dense(hidden_states)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update = self._key_dense(hidden_states)
                value_update = self._value_dense(hidden_states)
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
            key = self._key_dense(hidden_states)
            value = self._value_dense(hidden_states)

        query = self._rotary_embedding_layer(query)
        key = self._rotary_embedding_layer(key)

        key = ops.tile(key, [1, 1, self.num_key_value_groups, 1])
        value = ops.tile(value, [1, 1, self.num_key_value_groups, 1])

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask
        )

        attention_output_shape = ops.shape(attention_output)

        attention_output = ops.reshape(
            attention_output,
            [
                attention_output_shape[0],
                attention_output_shape[1],
                self.hidden_dim,
            ],
        )

        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return (attention_output, cache)
        return attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            mask_expansion_axis = -3
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(self, query, key, value, attention_mask=None):
        attention_scores = ops.einsum("aecd,abcd->acbe", key, query)

        norm_factor = ops.sqrt(
            ops.convert_to_tensor(self.attn_head_size, self.compute_dtype)
        )

        attention_scores /= norm_factor
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(
            "acbe,aecd->abcd", attention_scores, value
        )

        return attention_output, attention_scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "hidden_dim": self.hidden_dim,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
