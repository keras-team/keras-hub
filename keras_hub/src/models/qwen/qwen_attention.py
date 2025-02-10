import keras
from keras import ops

from keras_hub.src.models.llama.llama_attention import LlamaAttention
from keras_hub.src.utils.keras_utils import has_flash_attention_support


class Qwen2Attention(LlamaAttention):
    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1,
        kernel_initializer="glorot_uniform",
        dropout=0,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        **kwargs,
    ):
        super().__init__(
            num_query_heads,
            num_key_value_heads,
            rope_max_wavelength,
            rope_scaling_factor,
            kernel_initializer,
            dropout,
            **kwargs,
        )

        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size

    def _compute_attention(self, query, key, value, attention_mask=None):
        if has_flash_attention_support():
            # Use `dot_product_attention` with Flash Attention support if
            # available.
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            attention_output = ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )
            return attention_output

        attention_scores = ops.einsum(self._dot_product_equation, query, key)

        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        if self.use_sliding_window_attention:
            attention_mask = self._mask_sliding_window(
                attention_mask,
                cache_update_index=cache_update_index, 
                # TODO: cached attention for qwen
            )
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )

        return attention_output

    def _mask_sliding_window(
        self,
        attention_mask,
        cache_update_index=0,
    ):
        _, query_len, key_len = ops.shape(attention_mask)
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
