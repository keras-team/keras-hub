import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.multimodal_rotary_embedding import (
    MultimodalRotaryEmbedding,
)
from keras_hub.src.models.qwen3_moe.qwen3_moe_layernorm import Qwen3MoeLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


class Qwen3OmniAttention(keras.layers.Layer):
    """GQA attention with Multimodal RoPE and QK-norm for Qwen3-Omni.

    M-RoPE splits ``head_dim // 2`` into 3 sub-bands ``(t, h, w)``
    (defaults to ``(24, 20, 20)``); ``position_ids[0/1/2]`` feed the
    temporal / height / width channels respectively.

    Args:
        num_query_heads: int. Number of query heads.
        num_key_value_heads: int. Number of key/value heads (for GQA).
        head_dim: int. The dimension of each attention head.
        mrope_section: tuple of 3 ints. Dimension allocation for M-RoPE
            (text, temporal, spatial). Defaults to (24, 20, 20).
        rope_max_wavelength: int. Maximum wavelength for M-RoPE.
            Defaults to 1000000.
        rope_scaling_factor: float. Scaling factor for M-RoPE. Defaults to 1.0.
        rope_attention_scaling: float. Post-RoPE scaling applied to
            cos/sin embeddings. Defaults to 1.0.
        kernel_initializer: Initializer for kernel weights.
        dropout: float. Dropout rate for attention weights.
        layer_norm_epsilon: float. Epsilon for layer normalization.
        sliding_window_size: int or None. Size of sliding window.
            Defaults to None.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        head_dim=None,
        mrope_section=(24, 20, 20),
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        rope_attention_scaling=1.0,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        layer_norm_epsilon=1e-6,
        sliding_window_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.mrope_section = tuple(mrope_section)
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_attention_scaling = rope_attention_scaling
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.sliding_window_size = sliding_window_size

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        hidden_dim = inputs_shape[-1]
        if not self.head_dim:
            self.head_dim = hidden_dim // self.num_query_heads
        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        # Query projection with EinsumDense
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        # Query normalization (QK norm)
        self._query_dense_layer_norm = Qwen3MoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            head_dim=self.head_dim,
            name="query_dense_layernorm",
        )
        self._query_dense_layer_norm.build(inputs_shape)

        # Key projection
        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        # Key normalization (QK norm)
        self._key_dense_layer_norm = Qwen3MoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            head_dim=self.head_dim,
            name="key_dense_layernorm",
        )
        self._key_dense_layer_norm.build(inputs_shape)

        # Value projection
        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        # Softmax and dropout
        self._softmax = keras.layers.Softmax(
            axis=-1, dtype="float32", name="attention_softmax"
        )
        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy
        )

        # Output projection
        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

        # Multimodal RoPE
        self.multimodal_rotary_embedding = MultimodalRotaryEmbedding(
            mrope_section=self.mrope_section,
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            attention_scaling=self.rope_attention_scaling,
            dtype=self.dtype_policy,
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"
        self.built = True

    def call(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        if position_ids is None:
            positions = ops.repeat(
                ops.expand_dims(ops.arange(seq_len, dtype="int32"), 0),
                batch_size,
                axis=0,
            )
            position_ids = ops.stack([positions] * 3, axis=0)

        query = self._query_dense_layer_norm(self._query_dense(hidden_states))
        rotary = (
            self.multimodal_rotary_embedding.apply_multimodal_rotary_embedding
        )

        def _compute_key_value(x):
            key = self._key_dense_layer_norm(self._key_dense(x))
            return key, self._value_dense(x)

        if cache is not None:
            key_cache, value_cache = cache[:, 0, ...], cache[:, 1, ...]
            if cache_update_index is None:
                # Cached K/V already rotated; only rotate the new query.
                key, value = key_cache, value_cache
                query, _ = rotary(query, None, position_ids)
            else:
                key_update, value_update = _compute_key_value(hidden_states)
                query, key_update = rotary(query, key_update, position_ids)
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` must be `None` when `cache` is "
                    f"`None`; got cache_update_index={cache_update_index}."
                )
            key, value = _compute_key_value(hidden_states)
            query, key = rotary(query, key, position_ids)

        # GQA: tile KV heads up to query head count.
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._dropout_layer(
            self._compute_attention(
                query,
                key,
                value,
                attention_mask,
                cache_update_index=cache_update_index,
            ),
            training=training,
        )
        attention_output = self._output_dense(attention_output)
        return (
            (attention_output, cache) if cache is not None else attention_output
        )

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            return self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        return self._softmax(attention_scores)

    def _compute_attention(
        self, query, key, value, attention_mask=None, cache_update_index=None
    ):
        if self.sliding_window_size:
            if attention_mask is None:
                query_len = ops.shape(query)[1]
                key_len = ops.shape(key)[1]
                if cache_update_index is not None:
                    causal_mask = ops.cast(
                        ops.arange(key_len)
                        <= (cache_update_index + query_len - 1),
                        "bool",
                    )
                    attention_mask = ops.broadcast_to(
                        ops.reshape(causal_mask, (1, key_len)),
                        (query_len, key_len),
                    )
                else:
                    attention_mask = ops.tril(
                        ops.ones((query_len, key_len), dtype="bool")
                    )
                attention_mask = ops.expand_dims(attention_mask, 0)
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=cache_update_index or 0,
            )

        if fused_attention_op_available():
            if attention_mask is not None:
                attention_mask = ops.cast(
                    ops.expand_dims(attention_mask, axis=1), "bool"
                )
            return ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )

        attention_scores = ops.multiply(
            ops.einsum(self._dot_product_equation, query, key),
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        attention_scores = ops.cast(
            self._masked_softmax(attention_scores, attention_mask),
            self.compute_dtype,
        )
        return ops.einsum(self._combine_equation, attention_scores, value)

    def _mask_sliding_window(self, attention_mask, cache_update_index=0):
        _, query_len, key_len = ops.shape(attention_mask)
        all_ones = ops.ones((key_len, key_len), "bool")
        sliding_mask = ops.triu(
            all_ones, -self.sliding_window_size + 1
        ) * ops.tril(all_ones, self.sliding_window_size - 1)
        sliding_mask = ops.slice(
            sliding_mask, (cache_update_index, 0), (query_len, key_len)
        )
        return ops.logical_and(
            attention_mask, ops.cast(ops.expand_dims(sliding_mask, 0), "bool")
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "mrope_section": self.mrope_section,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_attention_scaling": self.rope_attention_scaling,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config
