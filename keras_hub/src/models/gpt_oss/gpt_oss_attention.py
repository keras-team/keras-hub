import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer


class CachedGptOssAttention(keras.layers.Layer):
    """A cached attention layer for GPT-OSS with sink tokens and sliding window.

    This layer implements the attention mechanism for the GPT-OSS model,
    including grouped query attention (GQA), rotary positional embeddings (RoPE),
    and a specific handling for "sink" tokens which are added to the attention
    logits before softmax. It also supports caching for efficient generation.

    Args:
        num_query_heads: Number of attention heads for queries.
        num_key_value_heads: Number of attention heads for keys and values.
            If `num_query_heads != num_key_value_heads`, grouped query attention
            is used.
        rope_max_wavelength: The maximum wavelength for the rotary embedding.
        rope_scaling_factor: Scaling factor for rotary embeddings.
        kernel_initializer: Initializer for the dense layer kernels.
        sliding_window: The size of the sliding window for attention.
            Tokens outside this window are masked. This parameter is used for
            configuration but the actual masking should be handled by the
            `attention_mask` input.
        dropout: Dropout rate for attention probabilities.
        use_bias: Whether to include bias terms in the dense projections.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        kernel_initializer="glorot_uniform",
        sliding_window=4096,  # Default from Qwen2/Mixtral, GptOss inherits from Qwen2Attention
        dropout=0,
        use_bias=False,  # From GptOssConfig.attention_bias
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.dropout = dropout
        self.use_bias = use_bias

        if self.num_query_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_query_heads ({self.num_query_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        self.num_key_value_groups = (
            self.num_query_heads // self.num_key_value_heads
        )
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

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

        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self._head_dim),
            kernel_initializer=self._kernel_initializer,
            use_bias=self.use_bias,
            dtype=self.dtype_policy,
            name="q_proj",
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
            use_bias=self.use_bias,
            dtype=self.dtype_policy,
            name="k_proj",
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
            use_bias=self.use_bias,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.value_dense.build(inputs_shape)

        # Sinks parameter: (num_attention_heads,)
        # PyTorch GptOssPreTrainedModel._init_weights initializes sinks with normal_
        # Using 0.02 as a common default stddev for normal init if _kernel_initializer doesn't have it
        stddev = (
            self._kernel_initializer.stddev
            if hasattr(self._kernel_initializer, "stddev")
            else 0.02
        )
        self.sinks = self.add_weight(
            name="sinks",
            shape=(self.num_query_heads,),
            initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=stddev
            ),
            dtype=self.dtype_policy,
        )

        self.softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",  # Softmax usually computed in float32 for stability
            name="attention_softmax",
        )

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self.output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, self._hidden_dim),
            kernel_initializer=self._kernel_initializer,
            use_bias=self.use_bias,
            dtype=self.dtype_policy,
            name="o_proj",
        )
        self.output_dense.build(
            (None, None, self.num_query_heads, self._head_dim)
        )

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
                # The cache has shape (batch, 2, seq_len, num_heads, head_dim)
                # key_update/value_update has shape (batch, new_seq_len, num_heads, head_dim)
                # We need to slice update at cache_update_index
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

        # Grouped Query Attention: repeat key and value heads if num_query_heads > num_key_value_heads
        # [batch_shape, seq_len, num_key_value_heads, head_dim]
        # -> [batch_shape, seq_len, num_heads, head_dim]
        if self.num_key_value_groups > 1:
            key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
            value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query, key, value, attention_mask, training=training
        )

        attention_output = self.dropout_layer(
            attention_output, training=training
        )

        attention_output = self.output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _use_fused_attention_op(self):
        # GPT-OSS attention includes "sink" tokens which are added to the logits
        # before softmax. The Keras `ops.dot_product_attention` does not support
        # this custom modification to the logits. Therefore, we must use the
        # manual attention calculation path.
        return False

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        # The _use_fused_attention_op is explicitly False for GptOssAttention
        # due to the sink token mechanism.

        # 1. Calculate raw attention scores
        attention_scores = ops.einsum(self._dot_product_equation, query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )

        # 2. Apply attention mask (if any)
        if attention_mask is not None:
            # attention_mask is typically (batch, 1, query_len, key_len) or (batch, query_len, key_len)
            # Expand mask to (batch, num_heads, query_len, key_len) if needed
            if ops.ndim(attention_mask) == 3:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
            attention_scores = attention_scores + attention_mask

        # 3. Prepare and concatenate sink tokens
        # sinks shape: (num_query_heads,)
        # Expand to (1, num_query_heads, 1, 1) then broadcast to (batch, num_query_heads, query_len, 1)
        sinks_expanded = ops.reshape(
            self.sinks, (1, self.num_query_heads, 1, 1)
        )
        # The attention_scores shape is (batch, num_heads, query_len, key_len)
        # We need to broadcast sinks_expanded to match batch, num_heads, query_len, and add a new last dim of 1
        sinks_expanded = ops.broadcast_to(
            sinks_expanded, ops.shape(attention_scores)[:-1] + (1,)
        )

        # Concatenate attention scores with sinks along the last dimension
        # Resulting shape: (batch, num_query_heads, query_len, key_len + 1)
        combined_logits = ops.concatenate(
            [attention_scores, sinks_expanded], axis=-1
        )

        # 4. Apply numerical stability clamping before softmax
        # combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        max_logits = ops.max(combined_logits, axis=-1, keepdims=True)
        combined_logits = combined_logits - max_logits

        # 5. Apply softmax
        # Softmax is applied to the combined logits (scores + sinks)
        probs = self.softmax(combined_logits)  # self.softmax is float32

        # 6. Drop the sink token probability to get final attention weights
        # scores = probs[..., :-1]
        scores = ops.slice(
            probs,
            [0, 0, 0, 0],
            ops.shape(probs)[:-1] + (ops.shape(probs)[-1] - 1,),
        )

        # 7. Cast to compute_dtype (dropout is handled outside this method)
        attention_weights = ops.cast(scores, self.compute_dtype)

        # 8. Compute weighted sum of values
        attention_output = ops.einsum(
            self._combine_equation, attention_weights, value
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
                "use_bias": self.use_bias,
            }
        )
        return config
