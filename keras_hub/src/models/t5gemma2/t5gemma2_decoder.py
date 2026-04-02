import keras
from keras import ops

from keras_hub.src.models.gemma3.gemma3_layers import RMSNormalization
from keras_hub.src.models.t5gemma2.t5gemma2_attention import (
    T5Gemma2MergedAttention,
)
from keras_hub.src.models.t5gemma2.t5gemma2_layers import T5Gemma2MLP


class T5Gemma2DecoderLayer(keras.layers.Layer):
    """Decoder layer for the T5Gemma2 model.

    This layer implements a single decoder block in the T5Gemma2
    architecture. Unlike T5Gemma which has separate self-attention and
    cross-attention sub-layers, T5Gemma2 uses a single
    `T5Gemma2MergedAttention` layer that fuses self-attention and
    cross-attention by concatenating their K/V pairs.

    Args:
        hidden_size: int, Dimensionality of hidden states.
        rms_norm_eps: float, Epsilon for RMS normalization.
        num_attention_heads: int, Number of attention heads.
        num_key_value_heads: int, Number of key-value heads for GQA.
        query_pre_attn_scalar: float, Scalar for query normalization.
        attention_bias: bool, Whether to include bias.
        intermediate_size: int, Intermediate size of the FFN.
        hidden_activation: str, Activation function for the FFN.
        dropout_rate: float, Dropout rate.
        head_dim: int, Dimensionality of each attention head.
        initializer_range: float, Range for the initializer.
        attention_dropout: float, Dropout for attention weights.
        layer_type: str, Either `"full_attention"` or
            `"sliding_attention"`.
        cross_attention_hidden_size: int, optional, Hidden size for
            cross-attention. Defaults to `hidden_size`.
        attn_logit_softcapping: float, optional, Softcapping value.
        sliding_window: int, optional, Window size for sliding
            attention.
        rope_max_wavelength: float, Maximum wavelength for RoPE.
        use_query_key_norm: bool, Whether to apply Q/K norm.
            Defaults to `True`.
        dtype: The dtype for computations. Defaults to `None`.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        hidden_size,
        rms_norm_eps,
        num_attention_heads,
        num_key_value_heads,
        query_pre_attn_scalar,
        attention_bias,
        intermediate_size,
        hidden_activation,
        dropout_rate,
        head_dim,
        initializer_range,
        attention_dropout,
        layer_type,
        cross_attention_hidden_size=None,
        attn_logit_softcapping=None,
        sliding_window=None,
        rope_max_wavelength=10000.0,
        rope_scaling_factor=1.0,
        use_query_key_norm=True,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.attention_type = layer_type
        self.sliding_window = sliding_window
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.attn_logit_softcapping = attn_logit_softcapping
        self.use_query_key_norm = use_query_key_norm

        if (
            self.attention_type == "sliding_attention"
            and self.sliding_window is None
        ):
            raise ValueError(
                "`sliding_window` must be set for `sliding_attention` "
                "layer type."
            )

        # Merged self+cross attention.
        self.merged_attn = T5Gemma2MergedAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            query_pre_attn_scalar=query_pre_attn_scalar,
            attention_bias=attention_bias,
            head_dim=self.head_dim,
            cross_attention_hidden_size=(
                cross_attention_hidden_size or hidden_size
            ),
            initializer_range=initializer_range,
            attention_dropout=attention_dropout,
            attn_logit_softcapping=attn_logit_softcapping,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            use_query_key_norm=use_query_key_norm,
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype_policy,
            name="merged_attention",
        )
        self.pre_self_attn_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="decoder_pre_self_attention_layernorm",
        )
        self.post_self_attn_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="decoder_post_self_attention_layernorm",
        )

        # MLP.
        self.mlp = T5Gemma2MLP(
            hidden_size,
            intermediate_size,
            hidden_activation,
            dropout_rate,
            initializer_range=initializer_range,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.pre_feedforward_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="decoder_pre_feedforward_layernorm",
        )
        self.post_feedforward_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="decoder_post_feedforward_layernorm",
        )

        self.dropout = keras.layers.Dropout(
            dropout_rate,
            dtype=self.dtype_policy,
            name="decoder_residual_dropout",
        )

    def build(self, input_shape):
        hidden_states_shape, encoder_hidden_states_shape = input_shape
        self.pre_self_attn_layernorm.build(hidden_states_shape)
        self.merged_attn.build(
            [hidden_states_shape, encoder_hidden_states_shape]
        )
        attn_output_shape, _ = self.merged_attn.compute_output_shape(
            [hidden_states_shape, encoder_hidden_states_shape]
        )
        self.post_self_attn_layernorm.build(attn_output_shape)
        self.dropout.build(attn_output_shape)
        self.pre_feedforward_layernorm.build(attn_output_shape)
        self.mlp.build(attn_output_shape)
        mlp_output_shape = self.mlp.compute_output_shape(attn_output_shape)
        self.post_feedforward_layernorm.build(mlp_output_shape)
        self.built = True

    def _make_causal_mask(
        self,
        hidden_states,
        padding_mask,
        cache=None,
        cache_update_index=None,
    ):
        """Creates a causal attention mask for self-attention."""
        if cache is not None:
            q_len = ops.shape(hidden_states)[1]
            kv_len = ops.shape(cache)[2]
            q_indices = ops.arange(0, q_len, dtype="int32") + cache_update_index
            kv_indices = ops.arange(0, kv_len, dtype="int32")
        else:
            q_len = kv_len = ops.shape(hidden_states)[1]
            q_indices = ops.arange(0, q_len, dtype="int32")
            kv_indices = ops.arange(0, kv_len, dtype="int32")
        causal_mask = kv_indices[None, :] <= q_indices[:, None]
        if self.attention_type == "sliding_attention":
            sliding_mask = (
                q_indices[:, None] - (self.sliding_window - 1)
            ) <= kv_indices[None, :]
            causal_mask = ops.logical_and(causal_mask, sliding_mask)
        final_mask = causal_mask[None, None, :, :]

        # Broadcast the mask to match the batch size.
        batch_size = ops.shape(hidden_states)[0]
        final_mask = ops.broadcast_to(
            final_mask, (batch_size, 1, q_len, kv_len)
        )

        if padding_mask is not None:
            padding_mask_slice = padding_mask[:, :kv_len]
            padding_mask_4d = padding_mask_slice[:, None, None, :]
            final_mask = ops.logical_and(final_mask, padding_mask_4d)
        return (1.0 - ops.cast(final_mask, hidden_states.dtype)) * -1e9

    def _make_cross_attention_mask(self, hidden_states, padding_mask):
        """Creates a bidirectional mask for cross-attention."""
        if padding_mask is None:
            return None
        q_len = ops.shape(hidden_states)[1]
        bidirectional_mask = padding_mask[:, None, None, :]
        # Broadcast to (batch, 1, q_len, enc_len).
        bidirectional_mask = ops.broadcast_to(
            bidirectional_mask,
            (
                ops.shape(hidden_states)[0],
                1,
                q_len,
                ops.shape(padding_mask)[1],
            ),
        )
        additive_mask = (
            1.0 - ops.cast(bidirectional_mask, hidden_states.dtype)
        ) * -1e9
        return additive_mask

    def call(
        self,
        inputs,
        self_attention_padding_mask=None,
        cross_attention_padding_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        """Forward pass of the decoder layer.

        Args:
            inputs: Tuple of (hidden_states, encoder_hidden_states).
            self_attention_padding_mask: Padding mask for decoder
                tokens.
            cross_attention_padding_mask: Padding mask for encoder
                tokens.
            cache: Tuple of (self_attn_cache, cross_attn_cache).
            cache_update_index: int, current position for caching.
            training: bool, training mode.

        Returns:
            Tuple of (hidden_states, updated_cache).
        """
        hidden_states, encoder_hidden_states = inputs
        self_attention_cache, cross_attention_cache = (
            cache if cache is not None else (None, None)
        )

        # Build the merged attention mask.
        self_attention_mask = self._make_causal_mask(
            hidden_states,
            self_attention_padding_mask,
            cache=self_attention_cache,
            cache_update_index=cache_update_index,
        )
        cross_attention_mask = self._make_cross_attention_mask(
            hidden_states, cross_attention_padding_mask
        )

        # Concatenate self and cross masks along the KV dimension.
        if cross_attention_mask is not None:
            merged_mask = ops.concatenate(
                [self_attention_mask, cross_attention_mask], axis=-1
            )
        else:
            merged_mask = self_attention_mask

        # Merged attention: self + cross.
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, updated_cache = self.merged_attn(
            inputs=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=merged_mask,
            cache=(self_attention_cache, cross_attention_cache),
            cache_update_index=cache_update_index,
            training=training,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(
            hidden_states, training=training
        )

        # MLP.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(
            hidden_states, training=training
        )
        return hidden_states, updated_cache

    def compute_output_shape(self, input_shape):
        hidden_states_shape, encoder_hidden_states_shape = input_shape
        batch_size, dec_seq_len, _ = hidden_states_shape
        _, enc_seq_len, _ = encoder_hidden_states_shape
        self_cache_shape = (
            batch_size,
            2,
            dec_seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        cross_cache_shape = (
            batch_size,
            2,
            enc_seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        return hidden_states_shape, (self_cache_shape, cross_cache_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "rms_norm_eps": self.rms_norm_eps,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "attention_bias": self.attention_bias,
                "intermediate_size": self.intermediate_size,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "initializer_range": self.initializer_range,
                "attention_dropout": self.attention_dropout,
                "layer_type": self.attention_type,
                "sliding_window": self.sliding_window,
                "rope_max_wavelength": self.rope_max_wavelength,
                "head_dim": self.head_dim,
                "cross_attention_hidden_size": (
                    self.cross_attention_hidden_size
                ),
                "attn_logit_softcapping": self.attn_logit_softcapping,
                "use_query_key_norm": self.use_query_key_norm,
            }
        )
        return config
