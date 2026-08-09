import keras

from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.models.t5gemma.t5gemma_attention import T5GemmaAttention
from keras_hub.src.models.t5gemma.t5gemma_layers import T5GemmaMLP


class T5GemmaDecoderLayer(keras.layers.Layer):
    """Decoder layer for the T5Gemma model.

    This layer implements a single decoder block in the T5Gemma architecture,
    comprising self-attention, cross-attention, and a feed-forward network
    (MLP).

    Args:
        hidden_size: int, The dimensionality of the hidden states.
        rms_norm_eps: float, The epsilon value for RMS normalization.
        num_attention_heads: int, The number of attention heads in
            self-attention and cross-attention.
        num_key_value_heads: int, The number of key-value heads for grouped
            query attention.
        query_pre_attn_scalar: float, Scalar to multiply queries by before
            attention.
        attention_bias: bool, Whether to include bias in attention computations.
        intermediate_size: int, The intermediate size of the feed-forward
            network.
        hidden_activation: str, The activation function used in the feed-forward
            network.
        dropout_rate: float, The dropout rate applied after attention and MLP.
        head_dim: int, The dimensionality of each attention head.
        initializer_range: float, The range for the random normal initializer.
        attention_dropout: float, The dropout rate applied to attention weights.
        layer_type: str, Type of attention layer, e.g., `"sliding_attention"`.
        cross_attention_hidden_size: int, optional, The hidden size for
            cross-attention. If None, it defaults to `hidden_size`. Defaults to
            `None`.
        attn_logit_softcapping: float, optional, The softcapping value for
            attention logits. Defaults to `None`.
        sliding_window: int, optional, The window size for sliding attention.
            Required if `layer_type` is `"sliding_attention"`. Defaults to
            `None`.
        rope_max_wavelength: float, The maximum wavelength for Rotary
            Positional Embeddings. Defaults to `10000.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
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
        self.layer_type = layer_type
        self.sliding_window = sliding_window
        self.rope_max_wavelength = rope_max_wavelength
        self.cross_attention_hidden_size = cross_attention_hidden_size
        self.attn_logit_softcapping = attn_logit_softcapping
        if (
            self.layer_type == "sliding_attention"
            and self.sliding_window is None
        ):
            raise ValueError(
                "`sliding_window` must be set for `sliding_attention` layer "
                "type."
            )

        # Self-attention.
        self.self_attn = T5GemmaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            query_pre_attn_scalar=query_pre_attn_scalar,
            attention_bias=attention_bias,
            head_dim=self.head_dim,
            attention_type="self",
            initializer_range=initializer_range,
            attention_dropout=attention_dropout,
            attn_logit_softcapping=attn_logit_softcapping,
            rope_max_wavelength=self.rope_max_wavelength,
            dtype=self.dtype_policy,
            name="self_attention",
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

        # Cross-attention.
        self.cross_attn = T5GemmaAttention(
            hidden_size=hidden_size,
            cross_attention_hidden_size=cross_attention_hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            query_pre_attn_scalar=query_pre_attn_scalar,
            attention_bias=attention_bias,
            head_dim=self.head_dim,
            attention_type="cross",
            initializer_range=initializer_range,
            attention_dropout=attention_dropout,
            attn_logit_softcapping=attn_logit_softcapping,
            dtype=self.dtype_policy,
            name="cross_attention",
        )
        self.pre_cross_attn_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="decoder_pre_cross_attention_layernorm",
        )
        self.post_cross_attn_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="decoder_post_cross_attention_layernorm",
        )

        # MLP.
        self.mlp = T5GemmaMLP(
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
        current_shape = hidden_states_shape
        self.self_attn.build(current_shape)
        attn_output_shape, _ = self.self_attn.compute_output_shape(
            current_shape
        )
        self.post_self_attn_layernorm.build(attn_output_shape)
        current_shape = attn_output_shape
        self.dropout.build(current_shape)
        self.pre_cross_attn_layernorm.build(current_shape)
        self.cross_attn.build([current_shape, encoder_hidden_states_shape])
        attn_output_shape, _ = self.cross_attn.compute_output_shape(
            [current_shape, encoder_hidden_states_shape]
        )
        self.post_cross_attn_layernorm.build(attn_output_shape)
        current_shape = attn_output_shape
        self.pre_feedforward_layernorm.build(current_shape)
        self.mlp.build(current_shape)
        mlp_output_shape = self.mlp.compute_output_shape(current_shape)
        self.post_feedforward_layernorm.build(mlp_output_shape)
        self.built = True

    def _make_self_attention_mask(
        self,
        hidden_states,
        padding_mask,
        cache=None,
        cache_update_index=None,
    ):
        if cache is not None:
            q_len = keras.ops.shape(hidden_states)[1]
            kv_len = keras.ops.shape(cache)[2]
            q_indices = (
                keras.ops.arange(0, q_len, dtype="int32") + cache_update_index
            )
            kv_indices = keras.ops.arange(0, kv_len, dtype="int32")
        else:
            q_len = kv_len = keras.ops.shape(hidden_states)[1]
            q_indices = keras.ops.arange(0, q_len, dtype="int32")
            kv_indices = keras.ops.arange(0, kv_len, dtype="int32")
        # Create the causal mask.
        causal_mask = kv_indices[None, :] <= q_indices[:, None]
        # Apply sliding window if applicable.
        if self.layer_type == "sliding_attention":
            sliding_mask = (
                q_indices[:, None] - self.sliding_window
            ) <= kv_indices[None, :]
            causal_mask = keras.ops.logical_and(causal_mask, sliding_mask)
        # Combine with padding mask.
        final_mask = causal_mask[None, None, :, :]
        if padding_mask is not None:
            padding_mask_slice = padding_mask[:, :kv_len]
            padding_mask_4d = padding_mask_slice[:, None, None, :]
            final_mask = keras.ops.logical_and(final_mask, padding_mask_4d)
        return (1.0 - keras.ops.cast(final_mask, hidden_states.dtype)) * -1e9

    def _make_cross_attention_mask(self, hidden_states, padding_mask):
        if padding_mask is None:
            return None
        bidirectional_mask = padding_mask[:, None, None, :]
        additive_bidirectional_mask = (
            1.0 - keras.ops.cast(bidirectional_mask, hidden_states.dtype)
        ) * -1e9
        return additive_bidirectional_mask

    def call(
        self,
        inputs,
        self_attention_padding_mask=None,
        cross_attention_padding_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        hidden_states, encoder_hidden_states = inputs
        self_attention_cache, cross_attention_cache = (
            cache if cache is not None else (None, None)
        )
        # Self Attention.
        residual = hidden_states
        self_attention_mask = self._make_self_attention_mask(
            hidden_states,
            self_attention_padding_mask,
            cache=self_attention_cache,
            cache_update_index=cache_update_index,
        )
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, updated_self_attention_cache = self.self_attn(
            inputs=hidden_states,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=cache_update_index,
            training=training,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(
            hidden_states, training=training
        )

        # Cross Attention.
        residual = hidden_states
        cross_attention_mask = self._make_cross_attention_mask(
            encoder_hidden_states, cross_attention_padding_mask
        )
        hidden_states = self.pre_cross_attn_layernorm(hidden_states)
        hidden_states, updated_cross_attention_cache = self.cross_attn(
            inputs=[hidden_states, encoder_hidden_states],
            attention_mask=cross_attention_mask,
            cache=cross_attention_cache,
            training=training,
        )

        hidden_states = self.post_cross_attn_layernorm(hidden_states)
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
        updated_cache = (
            updated_self_attention_cache,
            updated_cross_attention_cache,
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
                "layer_type": self.layer_type,
                "sliding_window": self.sliding_window,
                "rope_max_wavelength": self.rope_max_wavelength,
                "head_dim": self.head_dim,
                "cross_attention_hidden_size": self.cross_attention_hidden_size,
                "attn_logit_softcapping": self.attn_logit_softcapping,
            }
        )
        return config
