import inspect

import keras

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma.gemma_attention import CachedGemmaAttention
from keras_hub.src.models.t5gemma.t5gemma_layers import (
    t5gemma_kernel_initializer,
)
from keras_hub.src.utils.keras_utils import clone_initializer


def repeat_kv(hidden_states, n_rep):
    """Repeats the key/value hidden states to match the number of query heads
    for Grouped Query Attention (GQA).

    This function is used in `T5GemmaAttention` to broadcast key and value
    states across multiple query heads when Grouped Query Attention (GQA) is
    used (i.e., when `num_query_heads` > `num_key_value_heads`).

    Args:
        hidden_states: Tensor, The key or value hidden states with shape
            `(batch, sequence_length, num_key_value_heads, head_dim)`.
        n_rep: int, The number of times to repeat the key/value heads. This is
            typically `num_query_heads // num_key_value_heads`.

    Returns:
        Tensor: The expanded key/value hidden states with shape
            `(batch, sequence_length, num_query_heads, head_dim)`.
    """
    if n_rep == 1:
        return hidden_states
    batch, slen, num_key_value_heads, head_dim = keras.ops.shape(hidden_states)
    hidden_states = keras.ops.expand_dims(hidden_states, 3)
    hidden_states = keras.ops.tile(hidden_states, (1, 1, 1, n_rep, 1))
    return keras.ops.reshape(
        hidden_states, (batch, slen, num_key_value_heads * n_rep, head_dim)
    )


class T5GemmaAttention(CachedGemmaAttention):
    """A unified attention layer for T5Gemma that handles both self-attention
    and cross-attention.

    This layer performs attention with optional Rotary Positional Embeddings
    (RoPE) and supports Grouped Query Attention (GQA). It is used in
    `T5GemmaEncoderLayer` and `T5GemmaDecoderLayer`.

    Args:
        hidden_size: int, The dimensionality of the hidden states.
        num_attention_heads: int, The number of attention heads.
        num_key_value_heads: int, The number of key-value heads. For GQA, this
            can be less than `num_attention_heads`.
        query_pre_attn_scalar: float, Scalar to multiply queries by before
            attention.
        attention_bias: bool, Whether to include bias in the dense layers.
        head_dim: int, The dimensionality of each attention head.
        attention_type: str, The type of attention, either 'self' or 'cross'.
            Defaults to 'self'.
        cross_attention_hidden_size: int, optional, The dimensionality of
            encoder hidden states for cross-attention. Defaults to `None`.
        initializer_range: float, The range for the random normal initializer
            for kernel weights. Defaults to `0.02`.
        attention_dropout: float, The dropout rate applied to attention weights.
            Defaults to `0.0`.
        attn_logit_softcapping: float, optional, The softcapping value for
            attention logits. Defaults to `None`.
        rope_max_wavelength: float, The maximum wavelength for Rotary Positional
            Embeddings. Defaults to `10000.0`. Only used for self-attention.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        query_pre_attn_scalar,
        attention_bias,
        head_dim,
        attention_type="self",
        cross_attention_hidden_size=None,
        initializer_range=0.02,
        attention_dropout=0.0,
        attn_logit_softcapping=None,
        rope_max_wavelength=10000.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            head_dim=head_dim,
            num_query_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            kernel_initializer=t5gemma_kernel_initializer(initializer_range),
            logit_soft_cap=attn_logit_softcapping,
            dropout=attention_dropout,
            query_head_dim_normalize=False,
            use_sliding_window_attention=False,
            dtype=dtype,
            **kwargs,
        )
        if attention_type not in ["self", "cross"]:
            raise ValueError(
                f"attention_type must be 'self' or 'cross', but got "
                f"{attention_type}"
            )
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.cross_attention_hidden_size = (
            cross_attention_hidden_size or hidden_size
        )
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.rope_max_wavelength = rope_max_wavelength
        self.num_key_value_groups = (
            self.num_query_heads // self.num_key_value_heads
        )
        self.scaling = self.query_pre_attn_scalar**-0.5
        if self.attention_type == "self":
            self.rotary_embedding = RotaryEmbedding(
                max_wavelength=self.rope_max_wavelength,
                sequence_axis=1,
                feature_axis=3,
                name="rotary_embedding",
                dtype=self.dtype_policy,
            )

    def build(self, input_shape):
        self._kernel_initializer = t5gemma_kernel_initializer(
            self.initializer_range
        )

        if self.attention_type == "cross":
            hidden_states_shape, kv_states_shape = input_shape
        else:
            hidden_states_shape = input_shape
            kv_states_shape = input_shape
        # Query projection layer.
        self.hidden_dim = hidden_states_shape[-1]
        self.query_dense = keras.layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="nh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(hidden_states_shape)

        # Key projection layer.
        self.key_dense = keras.layers.EinsumDense(
            equation="bsd,dkh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="kh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(kv_states_shape)

        # Value projection layer.
        self.value_dense = keras.layers.EinsumDense(
            equation="bsd,dkh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="kh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(kv_states_shape)

        # Output projection layer.
        self.output_dense = keras.layers.EinsumDense(
            equation="btnh,nhd->btd",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="d" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (
                hidden_states_shape[0],
                hidden_states_shape[1],
                self.num_query_heads,
                self.head_dim,
            )
        )
        self.dropout_layer = keras.layers.Dropout(
            rate=self.attention_dropout,
            dtype=self.dtype_policy,
        )
        self.softmax = keras.layers.Softmax(axis=-1, dtype="float32")
        self.built = True

    def _compute_attention_without_fused_op(
        self, query_states, key_states, value_states, attention_mask, training
    ):
        attn_weights = keras.ops.einsum(
            "btnh,bsnh->bnts", query_states, key_states
        )
        attn_weights *= self.scaling
        if self.logit_soft_cap is not None:
            attn_weights = attn_weights / self.logit_soft_cap
            attn_weights = keras.ops.tanh(attn_weights)
            attn_weights = attn_weights * self.logit_soft_cap
        if attention_mask is not None:
            attn_weights += attention_mask
        attn_weights = keras.ops.cast(
            self.softmax(attn_weights),
            query_states.dtype,
        )
        attn_weights = self.dropout_layer(attn_weights, training=training)
        attn_output = keras.ops.einsum(
            "bnts,bsnh->btnh", attn_weights, value_states
        )
        return attn_output

    def _compute_attention(
        self, query_states, key_states, value_states, attention_mask, training
    ):
        if self._use_fused_attention_op():
            kwargs = {"bias": attention_mask}
            if self.logit_soft_cap is not None:
                sig = inspect.signature(keras.ops.dot_product_attention)
                # This is only supported in JAX TPU backend.
                # https://keras.io/api/ops/nn/#dot_product_attention-function
                if "attn_logits_soft_cap" in sig.parameters:
                    kwargs["attn_logits_soft_cap"] = self.logit_soft_cap
            return keras.ops.dot_product_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                scale=self.scaling,
                **kwargs,
            )
        return self._compute_attention_without_fused_op(
            query_states,
            key_states,
            value_states,
            attention_mask,
            training,
        )

    def call(
        self,
        inputs,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        if self.attention_type == "cross":
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
                raise ValueError(
                    "For cross-attention, `inputs` must be a list or tuple of "
                    "two tensors: `[hidden_states, encoder_hidden_states]`."
                )
            hidden_states, kv_states = inputs
            query_states = self.query_dense(hidden_states)
            if cache is not None:
                if cache_update_index is not None:
                    raise ValueError(
                        "`cache_update_index` should not be set for "
                        "cross-attention caching."
                    )
                key_states, value_states = cache[:, 0, ...], cache[:, 1, ...]
                updated_cache = cache
            else:
                key_states = self.key_dense(kv_states)
                value_states = self.value_dense(kv_states)
                updated_cache = keras.ops.stack(
                    (key_states, value_states), axis=1
                )
            # Repeat key-value heads for GQA.
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = self._compute_attention(
                query_states, key_states, value_states, attention_mask, training
            )
            attn_output = self.output_dense(attn_output)
            return attn_output, updated_cache
        else:  # Self-attention
            hidden_states = inputs
            kv_states = hidden_states
            query_states = self.query_dense(hidden_states)
            key_states = self.key_dense(kv_states)
            value_states = self.value_dense(kv_states)
            start_index = (
                0 if cache_update_index is None else cache_update_index
            )
            query_states = self.rotary_embedding(
                query_states, start_index=start_index
            )
            key_states = self.rotary_embedding(
                key_states, start_index=start_index
            )
            if cache is not None:
                if cache_update_index is None:
                    raise ValueError(
                        "Both `cache` and `cache_update_index` must be passed "
                        "for self-attention caching."
                    )
                key_cache, value_cache = cache[:, 0, ...], cache[:, 1, ...]
                start = [0, cache_update_index, 0, 0]
                key_states = keras.ops.slice_update(
                    key_cache, start, key_states
                )
                value_states = keras.ops.slice_update(
                    value_cache, start, value_states
                )
                cache = keras.ops.stack((key_states, value_states), axis=1)
            elif cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    "`None`."
                )
            else:
                cache = keras.ops.stack((key_states, value_states), axis=1)

            # Repeat key-value heads for GQA.
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = self._compute_attention(
                query_states, key_states, value_states, attention_mask, training
            )
            attn_output = self.output_dense(attn_output)
            return attn_output, cache

    def compute_output_shape(self, input_shape):
        if self.attention_type == "cross":
            hidden_states_shape, kv_states_shape = input_shape
        else:
            hidden_states_shape = input_shape
            kv_states_shape = input_shape
        attn_output_shape = hidden_states_shape
        kv_len = kv_states_shape[1]
        cache_shape = (
            hidden_states_shape[0],  # batch
            2,  # key and value
            kv_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        return attn_output_shape, cache_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "head_dim": self.head_dim,
                "num_attention_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "attention_bias": self.attention_bias,
                "attention_type": self.attention_type,
                "cross_attention_hidden_size": self.cross_attention_hidden_size,
                "initializer_range": self.initializer_range,
                "attention_dropout": self.attention_dropout,
                "attn_logit_softcapping": self.logit_soft_cap,
                "rope_max_wavelength": self.rope_max_wavelength,
            }
        )
        return config
