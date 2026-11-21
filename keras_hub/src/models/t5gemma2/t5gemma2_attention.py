import inspect

import keras

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention
from keras_hub.src.models.gemma3.gemma3_layers import RMSNormalization
from keras_hub.src.models.t5gemma2.t5gemma2_layers import (
    t5gemma2_kernel_initializer,
)
from keras_hub.src.utils.keras_utils import clone_initializer


def repeat_kv(hidden_states, n_rep):
    """Repeats the key/value hidden states for Grouped Query Attention.

    Args:
        hidden_states: Tensor with shape
            `(batch, sequence_length, num_key_value_heads, head_dim)`.
        n_rep: int, number of times to repeat.

    Returns:
        Tensor with shape
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


class T5Gemma2Attention(CachedGemma3Attention):
    """Self-attention layer for T5Gemma2 encoder and decoder.

    This layer performs self-attention with Rotary Positional Embeddings
    (RoPE), optional Q/K normalization (Gemma3-style), and optional
    attention logit softcapping. Supports Grouped Query Attention (GQA).

    Used in `T5Gemma2EncoderLayer` for bidirectional self-attention
    and can also be used in the decoder for self-attention.

    Args:
        hidden_size: int, The dimensionality of the hidden states.
        num_attention_heads: int, The number of attention heads.
        num_key_value_heads: int, The number of key-value heads for GQA.
        query_pre_attn_scalar: float, Scalar to multiply queries by
            before attention.
        attention_bias: bool, Whether to include bias in dense layers.
        head_dim: int, The dimensionality of each attention head.
        initializer_range: float, The range for the initializer.
            Defaults to `0.02`.
        attention_dropout: float, Dropout rate for attention weights.
            Defaults to `0.0`.
        attn_logit_softcapping: float, optional, Softcapping value.
            Defaults to `None`.
        rope_max_wavelength: float, Maximum wavelength for RoPE.
            Defaults to `10000.0`.
        use_query_key_norm: bool, Whether to apply RMS normalization on
            query and key. Defaults to `True` (Gemma3-style).
        rms_norm_eps: float, Epsilon for RMS normalization.
            Defaults to `1e-6`.
        dtype: The dtype for computations and weights. Defaults to `None`.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        query_pre_attn_scalar,
        attention_bias,
        head_dim,
        initializer_range=0.02,
        attention_dropout=0.0,
        attn_logit_softcapping=None,
        rope_max_wavelength=10000.0,
        rope_scaling_factor=1.0,
        use_query_key_norm=True,
        rms_norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            head_dim=head_dim,
            num_query_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            kernel_initializer=t5gemma2_kernel_initializer(initializer_range),
            logit_soft_cap=attn_logit_softcapping,
            dropout=attention_dropout,
            query_head_dim_normalize=False,
            use_sliding_window_attention=False,
            dtype=dtype,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.use_query_key_norm = use_query_key_norm
        self.rms_norm_eps = rms_norm_eps
        self.num_key_value_groups = (
            self.num_query_heads // self.num_key_value_heads
        )
        self.scaling = self.query_pre_attn_scalar**-0.5

    def build(self, input_shape):
        self._kernel_initializer = t5gemma2_kernel_initializer(
            self.initializer_range
        )
        hidden_states_shape = input_shape
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

        self.key_dense = keras.layers.EinsumDense(
            equation="bsd,dkh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="kh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(hidden_states_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bsd,dkh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="kh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(hidden_states_shape)

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

        # Q/K normalization (Gemma3-style).
        if self.use_query_key_norm:
            self.query_norm = RMSNormalization(
                epsilon=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="query_norm",
            )
            self.query_norm.build(
                self.query_dense.compute_output_shape(hidden_states_shape)
            )
            self.key_norm = RMSNormalization(
                epsilon=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="key_norm",
            )
            self.key_norm.build(
                self.key_dense.compute_output_shape(hidden_states_shape)
            )

        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            sequence_axis=1,
            feature_axis=3,
            name="rotary_embedding",
            dtype=self.dtype_policy,
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
        hidden_states = inputs
        query_states = self.query_dense(hidden_states)
        key_states = self.key_dense(hidden_states)
        value_states = self.value_dense(hidden_states)

        # Apply Q/K normalization.
        if self.use_query_key_norm:
            query_states = self.query_norm(query_states)
            key_states = self.key_norm(key_states)

        # Apply RoPE.
        start_index = 0 if cache_update_index is None else cache_update_index
        query_states = self.rotary_embedding(
            query_states, start_index=start_index
        )
        key_states = self.rotary_embedding(key_states, start_index=start_index)

        # Handle caching for autoregressive generation.
        if cache is not None:
            if cache_update_index is None:
                raise ValueError(
                    "Both `cache` and `cache_update_index` must be "
                    "passed for self-attention caching."
                )
            key_cache, value_cache = cache[:, 0, ...], cache[:, 1, ...]
            start = [0, cache_update_index, 0, 0]
            key_states = keras.ops.slice_update(key_cache, start, key_states)
            value_states = keras.ops.slice_update(
                value_cache, start, value_states
            )
            cache = keras.ops.stack((key_states, value_states), axis=1)
        elif cache_update_index is not None:
            raise ValueError(
                "`cache_update_index` should not be set if `cache` is `None`."
            )
        else:
            cache = keras.ops.stack((key_states, value_states), axis=1)

        # Repeat K/V for GQA.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self._compute_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            training,
        )
        attn_output = self.output_dense(attn_output)
        return attn_output, cache

    def compute_output_shape(self, input_shape):
        hidden_states_shape = input_shape
        attn_output_shape = hidden_states_shape
        kv_len = hidden_states_shape[1]
        cache_shape = (
            hidden_states_shape[0],
            2,
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
                "initializer_range": self.initializer_range,
                "attention_dropout": self.attention_dropout,
                "attn_logit_softcapping": self.logit_soft_cap,
                "rope_max_wavelength": self.rope_max_wavelength,
                "use_query_key_norm": self.use_query_key_norm,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config


class T5Gemma2MergedAttention(CachedGemma3Attention):
    """Merged self-attention and cross-attention for T5Gemma2 decoder.

    This layer fuses self-attention and cross-attention into a single
    attention computation. The decoder's Q/K/V are computed from the
    decoder hidden states (self-attention), while additional K/V are
    computed from the encoder hidden states (cross-attention). The
    self-attention and cross-attention K/V are concatenated, and a
    single attention computation is performed over the merged K/V.

    This merged approach is the key architectural difference between
    T5Gemma2 and T5Gemma.

    Args:
        hidden_size: int, Dimensionality of the decoder hidden states.
        num_attention_heads: int, Number of attention heads.
        num_key_value_heads: int, Number of key-value heads for GQA.
        query_pre_attn_scalar: float, Scalar for query normalization.
        attention_bias: bool, Whether to include bias.
        head_dim: int, Dimensionality of each attention head.
        cross_attention_hidden_size: int, optional, Hidden size of the
            encoder states. Defaults to `hidden_size`.
        initializer_range: float, Range for the initializer.
            Defaults to `0.02`.
        attention_dropout: float, Dropout rate.
            Defaults to `0.0`.
        attn_logit_softcapping: float, optional, Softcapping value.
            Defaults to `None`.
        rope_max_wavelength: float, Maximum wavelength for RoPE.
            Defaults to `10000.0`.
        use_query_key_norm: bool, Whether to apply Q/K norm.
            Defaults to `True`.
        rms_norm_eps: float, Epsilon for RMS normalization.
            Defaults to `1e-6`.
        dtype: The dtype for computations. Defaults to `None`.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        query_pre_attn_scalar,
        attention_bias,
        head_dim,
        cross_attention_hidden_size=None,
        initializer_range=0.02,
        attention_dropout=0.0,
        attn_logit_softcapping=None,
        rope_max_wavelength=10000.0,
        rope_scaling_factor=1.0,
        use_query_key_norm=True,
        rms_norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            head_dim=head_dim,
            num_query_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            kernel_initializer=t5gemma2_kernel_initializer(initializer_range),
            logit_soft_cap=attn_logit_softcapping,
            dropout=attention_dropout,
            query_head_dim_normalize=False,
            use_sliding_window_attention=False,
            dtype=dtype,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.cross_attention_hidden_size = (
            cross_attention_hidden_size or hidden_size
        )
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.use_query_key_norm = use_query_key_norm
        self.rms_norm_eps = rms_norm_eps
        self.num_key_value_groups = (
            self.num_query_heads // self.num_key_value_heads
        )
        self.scaling = self.query_pre_attn_scalar**-0.5

    def build(self, input_shape):
        # Only decoder_shape needed; K/V projections are shared.
        if isinstance(input_shape, (list, tuple)):
            decoder_shape = input_shape[0]
        else:
            decoder_shape = input_shape

        self._kernel_initializer = t5gemma2_kernel_initializer(
            self.initializer_range
        )
        self.hidden_dim = decoder_shape[-1]

        # Q projection from decoder hidden states.
        self.query_dense = keras.layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="nh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(decoder_shape)

        # K/V projections shared for self-attn and cross-attn.
        self.key_dense = keras.layers.EinsumDense(
            equation="bsd,dkh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="kh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(decoder_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bsd,dkh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="kh" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(decoder_shape)

        # Output projection.
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
                decoder_shape[0],
                decoder_shape[1],
                self.num_query_heads,
                self.head_dim,
            )
        )

        # Q/K normalization (Gemma3-style).
        if self.use_query_key_norm:
            self.query_norm = RMSNormalization(
                epsilon=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="query_norm",
            )
            self.query_norm.build(
                self.query_dense.compute_output_shape(decoder_shape)
            )
            self.key_norm = RMSNormalization(
                epsilon=self.rms_norm_eps,
                dtype=self.dtype_policy,
                name="key_norm",
            )
            self.key_norm.build(
                self.key_dense.compute_output_shape(decoder_shape)
            )

        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            sequence_axis=1,
            feature_axis=3,
            name="rotary_embedding",
            dtype=self.dtype_policy,
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
        encoder_hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        """Forward pass of merged self+cross attention.

        Args:
            inputs: Decoder hidden states, shape
                `(batch, decoder_seq_len, hidden_dim)`.
            encoder_hidden_states: Encoder output, shape
                `(batch, encoder_seq_len, hidden_dim)`.
            attention_mask: Merged attention mask, shape
                `(batch, 1, decoder_seq_len, decoder_kv_len + encoder_len)`.
                This is the concatenation of the causal self-attention
                mask and the bidirectional cross-attention mask.
            cache: Tuple of (self_attn_cache, cross_attn_cache) or None.
            cache_update_index: int, the current position in the
                sequence for caching.
            training: bool, whether in training mode.

        Returns:
            Tuple of (attn_output, updated_cache).
        """
        hidden_states = inputs
        self_attention_cache, cross_attention_cache = (
            cache if cache is not None else (None, None)
        )

        # Self-attention Q/K/V from decoder hidden states.
        query_states = self.query_dense(hidden_states)
        key_states = self.key_dense(hidden_states)
        value_states = self.value_dense(hidden_states)

        # Apply Q/K normalization.
        if self.use_query_key_norm:
            query_states = self.query_norm(query_states)
            key_states = self.key_norm(key_states)

        # Apply RoPE to self-attention Q/K only (not cross-attention).
        start_index = 0 if cache_update_index is None else cache_update_index
        query_states = self.rotary_embedding(
            query_states, start_index=start_index
        )
        key_states = self.rotary_embedding(key_states, start_index=start_index)

        # Update self-attention cache.
        if self_attention_cache is not None:
            key_cache_self = self_attention_cache[:, 0, ...]
            value_cache_self = self_attention_cache[:, 1, ...]
            start = [0, cache_update_index, 0, 0]
            key_states = keras.ops.slice_update(
                key_cache_self, start, key_states
            )
            value_states = keras.ops.slice_update(
                value_cache_self, start, value_states
            )
            updated_self_cache = keras.ops.stack(
                (key_states, value_states), axis=1
            )
        else:
            updated_self_cache = keras.ops.stack(
                (key_states, value_states), axis=1
            )

        # Cross-attention K/V from encoder hidden states.
        if cross_attention_cache is not None:
            # Reuse cached encoder K/V.
            cross_key_states = cross_attention_cache[:, 0, ...]
            cross_value_states = cross_attention_cache[:, 1, ...]
            updated_cross_cache = cross_attention_cache
        else:
            cross_key_states = self.key_dense(encoder_hidden_states)
            cross_value_states = self.value_dense(encoder_hidden_states)
            # Apply K normalization to cross-attention keys.
            if self.use_query_key_norm:
                cross_key_states = self.key_norm(cross_key_states)
            updated_cross_cache = keras.ops.stack(
                (cross_key_states, cross_value_states), axis=1
            )

        # Merge self-attention and cross-attention K/V.
        merged_key_states = keras.ops.concatenate(
            [key_states, cross_key_states], axis=1
        )
        merged_value_states = keras.ops.concatenate(
            [value_states, cross_value_states], axis=1
        )

        # Repeat K/V for GQA.
        merged_key_states = repeat_kv(
            merged_key_states, self.num_key_value_groups
        )
        merged_value_states = repeat_kv(
            merged_value_states, self.num_key_value_groups
        )

        attn_output = self._compute_attention(
            query_states,
            merged_key_states,
            merged_value_states,
            attention_mask,
            training,
        )
        attn_output = self.output_dense(attn_output)
        updated_cache = (updated_self_cache, updated_cross_cache)
        return attn_output, updated_cache

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            decoder_shape, encoder_shape = input_shape
        else:
            decoder_shape = input_shape
            encoder_shape = input_shape
        attn_output_shape = decoder_shape
        dec_kv_len = decoder_shape[1]
        enc_kv_len = encoder_shape[1]
        self_cache_shape = (
            decoder_shape[0],
            2,
            dec_kv_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        cross_cache_shape = (
            decoder_shape[0],
            2,
            enc_kv_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        return attn_output_shape, (self_cache_shape, cross_cache_shape)

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
                "cross_attention_hidden_size": (
                    self.cross_attention_hidden_size
                ),
                "initializer_range": self.initializer_range,
                "attention_dropout": self.attention_dropout,
                "attn_logit_softcapping": self.logit_soft_cap,
                "rope_max_wavelength": self.rope_max_wavelength,
                "use_query_key_norm": self.use_query_key_norm,
                "rms_norm_eps": self.rms_norm_eps,
            }
        )
        return config
