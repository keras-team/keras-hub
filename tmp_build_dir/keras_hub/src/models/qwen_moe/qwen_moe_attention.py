import inspect
import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available
from keras_hub.src.utils.keras_utils import gpu_supports_fused_attention_op
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.keras_utils import running_on_tpu


class QwenMoeAttention(keras.layers.Layer):
    """A multi-head attention layer for Qwen-Moe model

    This attention implementation supports grouped-query attention (GQA) where
    the number of key-value heads can be less than the number of query heads.

    Args:
        num_query_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads (for GQA).
        rope_max_wavelength: Maximum wavelength for RoPE (Rotary Position
            Embedding).
        rope_scaling_factor: Scaling factor for RoPE, used for extending
            context length.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias weights.
        dropout: Dropout rate for attention weights.
        use_sliding_window_attention: Whether to use sliding window
            attention.
        sliding_window_size: Size of the sliding window for attention.
        **kwargs: Additional keyword arguments to pass to the Layer.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        dropout=0,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.dropout = dropout

        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength

        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.bias_initializer = keras.initializers.get(
            clone_initializer(bias_initializer)
        )

        self.rope_scaling_factor = rope_scaling_factor
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.logit_soft_cap = None

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
        head_dim = hidden_dim // self.num_query_heads
        self._inv_norm_factor = 1.0 / math.sqrt(head_dim)
        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            bias_axes="uh",
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            bias_axes="vh",
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            bias_axes="vh",
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build((None, None, self.num_query_heads, head_dim))

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            dtype=self.dtype_policy,
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
        """Applies attention mechanism to the input hidden states.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length,
                hidden_size].
            attention_mask: Mask tensor of shape [batch_size, seq_length,
                seq_length].
            cache: Optional cached key and value tensors.
            cache_update_index: Index at which to update the cache.
            training: Boolean indicating whether in training mode.

        Returns:
            attention_output: Output tensor after applying attention.
            cache: Updated cache tensors (if cache is provided).
        """
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self.query_dense(hidden_states)

        # Compute RoPE for queries
        query = self.rotary_embedding_layer(query, start_index=start_index)

        def _compute_key_value(x):
            key, value = self.key_dense(x), self.value_dense(x)
            # Compute RoPE for keys
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
            query,
            key,
            value,
            attention_mask,
            cache_update_index=cache_update_index,
        )

        attention_output = self._dropout_layer(
            attention_output, training=training
        )

        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        """Applies softmax with optional masking.

        Args:
            attention_scores: Attention score tensor.
            attention_mask: Optional mask tensor.

        Returns:
            Masked softmax attention weights.
        """
        if attention_mask is not None:
            return self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        return self._softmax(attention_scores)

    def _use_fused_attention_op(self):
        if not fused_attention_op_available():
            return False
        if self.dropout > 0.0:
            return False
        if running_on_gpu():
            return gpu_supports_fused_attention_op()
        elif running_on_tpu():
            # TPU supports softcap with on keras >= 3.10.
            sig = inspect.signature(ops.dot_product_attention)
            return "attn_logits_soft_cap" in sig.parameters
        else:
            return False

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        cache_update_index=None,
        **kwargs,
    ):
        """Computes attention using query, key, and value tensors.

        Uses Flash Attention when available for better performance.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Optional mask tensor.
            cache_update_index: Index for sliding window computation.

        Returns:
            attention_output: Output tensor after applying attention.
        """
        if self._use_fused_attention_op():
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")

            attention_output = ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
                **kwargs,
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
                cache_update_index=cache_update_index
                if cache_update_index
                else 0,
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
        """Creates and combines a sliding window mask with the attention mask.

        Args:
            attention_mask: Original attention mask.
            cache_update_index: Starting index for the sliding window.

        Returns:
            Combined attention mask with sliding window constraints.
        """
        _, query_len, key_len = ops.shape(attention_mask)
        # Compute the sliding window for square attention.
        all_ones = ops.ones((key_len, key_len), "bool")
        sliding_mask = ops.triu(
            all_ones, -1 * self.sliding_window_size + 1
        ) * ops.tril(all_ones, self.sliding_window_size - 1)
        # Slice the window for short queries during generation.
        start = (cache_update_index, 0)
        sliding_mask = ops.slice(sliding_mask, start, (query_len, key_len))
        sliding_mask = ops.expand_dims(sliding_mask, 0)
        return ops.logical_and(attention_mask, ops.cast(sliding_mask, "bool"))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "dropout": self.dropout,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config
