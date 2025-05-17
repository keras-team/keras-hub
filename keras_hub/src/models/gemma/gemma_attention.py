import inspect

import keras
import numpy as np
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import gpu_supports_fused_attention_op
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.keras_utils import running_on_tpu


class CachedGemmaAttention(keras.layers.Layer):
    """A cached grouped query attention layer."""

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        kernel_initializer="glorot_uniform",
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        query_head_dim_normalize=True,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.logit_soft_cap = logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.query_head_dim_normalize = query_head_dim_normalize
        self.dropout = dropout

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.query_head_dim_normalize = query_head_dim_normalize

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

        self.rope_layer = RotaryEmbedding(
            max_wavelength=10_000.0, dtype=self.dtype_policy
        )

        self.built = True

    def _apply_rope(self, x, start_index):
        """Rope rotate q or k."""
        x = self.rope_layer(x, start_index=start_index)
        # Gemma uses a different layout for positional embeddings.
        # The transformation below ensures the embeddings are numerically
        # equivalent to the original gemma implementation.
        x = ops.reshape(
            ops.stack(ops.split(x, 2, axis=-1), axis=-1), ops.shape(x)
        )
        return x

    def _use_fused_attention_op(self):
        if not fused_attention_op_available():
            return False
        if self.dropout > 0.0:
            return False
        if running_on_gpu():
            # GPU never supports softcap in the fused op.
            if self.logit_soft_cap is not None:
                return False
            return gpu_supports_fused_attention_op()
        elif running_on_tpu():
            # TPU supports softcap with on keras >= 3.10.
            sig = inspect.signature(ops.dot_product_attention)
            return "attn_logits_soft_cap" in sig.parameters
        else:
            return False

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
        cache_update_index=0,
    ):
        if self.query_head_dim_normalize:
            query_normalization = 1 / np.sqrt(self.head_dim)
        else:
            query_normalization = 1 / np.sqrt(
                self.hidden_dim // self.num_query_heads
            )

        if self.use_sliding_window_attention and attention_mask is not None:
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=cache_update_index,
            )

        if self._use_fused_attention_op():
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            # Only pass soft cap if needed as not all keras versions support.
            if self.logit_soft_cap:
                kwargs = {"attn_logits_soft_cap": self.logit_soft_cap}
            else:
                kwargs = {}
            return ops.dot_product_attention(
                query=q,
                key=k,
                value=v,
                mask=attention_mask,
                scale=query_normalization,
                **kwargs,
            )

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

        # Fallback to standard attention if flash attention is disabled
        attention_logits = ops.einsum("btkgh,bskh->bkgts", q, k)
        if self.logit_soft_cap is not None:
            attention_logits = ops.divide(attention_logits, self.logit_soft_cap)
            attention_logits = ops.multiply(
                ops.tanh(attention_logits), self.logit_soft_cap
            )

        if attention_mask is not None:
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

    def _mask_sliding_window(
        self,
        attention_mask,
        cache_update_index=0,
    ):
        batch_size, query_len, key_len = ops.shape(attention_mask)
        # Compute the sliding window for square attention.
        all_ones = ops.ones((key_len, key_len), "bool")
        if keras.config.backend() == "tensorflow":
            # TODO: trui/tril has issues with dynamic shape on the tensorflow
            # backend. We should fix, but use `band_part` for now.
            import tensorflow as tf

            band_size = ops.minimum(key_len, self.sliding_window_size - 1)
            band_size = ops.cast(band_size, "int32")
            sliding_mask = tf.linalg.band_part(all_ones, band_size, band_size)
        else:
            sliding_mask = ops.triu(
                all_ones, -1 * self.sliding_window_size + 1
            ) * ops.tril(all_ones, self.sliding_window_size - 1)
        # Slice the window for short queries during generation.
        start = (cache_update_index, 0)
        sliding_mask = ops.slice(sliding_mask, start, (query_len, key_len))
        sliding_mask = ops.expand_dims(sliding_mask, 0)
        return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))

    def call(
        self,
        x,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        training=False,
    ):
        query = self.query_dense(x)
        query = self._apply_rope(query, cache_update_index)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            key_update = self.key_dense(x)
            key_update = self._apply_rope(key_update, cache_update_index)
            value_update = self.value_dense(x)
            start = [0, cache_update_index, 0, 0]
            key = ops.slice_update(key_cache, start, key_update)
            value = ops.slice_update(value_cache, start, value_update)
            cache = ops.stack((key, value), axis=1)
        else:
            key = self.key_dense(x)
            key = self._apply_rope(key, cache_update_index)
            value = self.value_dense(x)

        attention_vec = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            training=training,
            cache_update_index=cache_update_index,
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

    def compute_output_shape(self, input_shape):
        return input_shape
