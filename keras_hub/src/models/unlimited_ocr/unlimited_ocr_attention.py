import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


class UnlimitedOCRAttention(keras.layers.Layer):
    """Self-attention with rotary embeddings, grouped heads and no bias.

    The layer computes multi-head attention. Each key and value head serves
    `num_query_heads // num_key_value_heads` query heads. The query and key
    tensors get rotary embeddings in the rotate-half layout. The layer
    supports a key/value cache for generation. The cache holds the key and
    value heads, not the repeated heads.

    Args:
        num_query_heads: int. The number of attention heads.
        num_key_value_heads: int or `None`. The number of key and value
            heads. It must divide `num_query_heads`. If `None`, it equals
            `num_query_heads`. Defaults to `None`.
        rope_max_wavelength: int. The maximum wavelength of the rotary
            embedding. Defaults to `10000`.
        kernel_initializer: string or `keras.initializers` initializer. The
            initializer of the projection kernels. Defaults to
            `"glorot_uniform"`.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads=None,
        rope_max_wavelength=10000,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_key_value_heads is None:
            num_key_value_heads = num_query_heads
        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "`num_key_value_heads` must divide `num_query_heads`. "
                f"Received: num_query_heads={num_query_heads}, "
                f"num_key_value_heads={num_key_value_heads}"
            )
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size, q = query length, k = key/value length,
        # m = model dim, u = number of heads, h = head dim.
        hidden_dim = inputs_shape[-1]
        head_dim = hidden_dim // self.num_query_heads
        self._inv_norm_factor = 1.0 / math.sqrt(head_dim)

        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            equation="bkm,muh->bkuh",
            output_shape=(None, self.num_key_value_heads, head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bkm,muh->bkuh",
            output_shape=(None, self.num_key_value_heads, head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(
            axis=-1, dtype="float32", name="attention_softmax"
        )

        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build((None, None, self.num_query_heads, head_dim))

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            dtype=self.dtype_policy,
        )
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        cache_write_mask=None,
    ):
        """Apply attention to the hidden states.

        Args:
            hidden_states: Tensor of shape `(batch, length, hidden_dim)`.
            attention_mask: Optional boolean tensor of shape
                `(batch, length, key_length)`.
            cache: Optional tensor of shape
                `(batch, 2, max_length, num_heads, head_dim)`. The tensor
                holds the cached keys and values.
            cache_update_index: Optional int. The position of the first
                token in `hidden_states`.
            cache_write_mask: Optional boolean tensor of shape `(batch,)`.
                A row with `False` keeps its cached keys and values at the
                updated positions.

        Returns:
            The attention output. When `cache` is set, the layer returns an
            `(output, cache)` tuple.
        """
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self.query_dense(hidden_states)
        query = self.rotary_embedding_layer(query, start_index=start_index)

        def _compute_key_value(x):
            key, value = self.key_dense(x), self.value_dense(x)
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
                if cache_write_mask is not None:
                    write = ops.reshape(cache_write_mask, (-1, 1, 1, 1))
                    shape = ops.shape(key_update)
                    old_key = ops.slice(key_cache, start, shape)
                    old_value = ops.slice(value_cache, start, shape)
                    key_update = ops.where(write, key_update, old_key)
                    value_update = ops.where(write, value_update, old_value)
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

        attention_output = self._compute_attention(
            query, key, value, attention_mask
        )
        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _compute_attention(self, query, key, value, attention_mask=None):
        if self.num_key_value_groups > 1:
            key = ops.repeat(key, self.num_key_value_groups, axis=2)
            value = ops.repeat(value, self.num_key_value_groups, axis=2)
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

        attention_scores = ops.einsum("bquh,bkuh->buqk", query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        if attention_mask is not None:
            attention_scores = self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        else:
            attention_scores = self._softmax(attention_scores)
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        return ops.einsum("buqk,bkuh->bquh", attention_scores, value)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
