import keras
from keras import ops

from keras_hub.src.models.gemma3.gemma3_layers import RMSNormalization
from keras_hub.src.models.t5gemma2.t5gemma2_attention import T5Gemma2Attention
from keras_hub.src.models.t5gemma2.t5gemma2_layers import T5Gemma2MLP


class T5Gemma2EncoderLayer(keras.layers.Layer):
    """Encoder layer for the T5Gemma2 model.

    This layer implements a single encoder block in the T5Gemma2
    architecture, comprising bidirectional self-attention and a
    feed-forward network (MLP). It uses Gemma3-style Q/K normalization.

    Each encoder layer has an `attention_type` attribute that specifies
    whether it uses `"full_attention"` or `"sliding_attention"`. The
    backbone uses this to route the correct RoPE embeddings and
    attention masks.

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
        initializer_range: float, Range for the initializer.
        attention_dropout: float, Dropout for attention weights.
        layer_type: str, Either `"full_attention"` or
            `"sliding_attention"`.
        head_dim: int, Dimensionality of each attention head.
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
        initializer_range,
        attention_dropout,
        layer_type,
        head_dim,
        attn_logit_softcapping=None,
        sliding_window=None,
        rope_max_wavelength=10000.0,
        rope_scaling_factor=1.0,
        use_query_key_norm=True,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
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
        self.head_dim = head_dim
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

        self.self_attn = T5Gemma2Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            query_pre_attn_scalar=query_pre_attn_scalar,
            attention_bias=attention_bias,
            head_dim=self.head_dim,
            initializer_range=initializer_range,
            attention_dropout=attention_dropout,
            attn_logit_softcapping=attn_logit_softcapping,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            use_query_key_norm=use_query_key_norm,
            rms_norm_eps=rms_norm_eps,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.pre_self_attn_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="pre_self_attention_layernorm",
        )
        self.post_self_attn_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="post_self_attention_layernorm",
        )

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
            name="pre_feedforward_layernorm",
        )
        self.post_feedforward_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="post_feedforward_layernorm",
        )
        self.dropout = keras.layers.Dropout(
            dropout_rate,
            dtype=self.dtype_policy,
            name="residual_dropout",
        )

    def build(self, input_shape):
        self.pre_self_attn_layernorm.build(input_shape)
        self.self_attn.build(input_shape)
        attn_output_shape, _ = self.self_attn.compute_output_shape(input_shape)
        self.post_self_attn_layernorm.build(attn_output_shape)
        self.dropout.build(attn_output_shape)
        self.pre_feedforward_layernorm.build(attn_output_shape)
        self.mlp.build(attn_output_shape)
        mlp_output_shape = self.mlp.compute_output_shape(attn_output_shape)
        self.post_feedforward_layernorm.build(mlp_output_shape)
        self.built = True

    def _make_attention_mask(self, hidden_states, padding_mask):
        attention_mask = padding_mask[:, None, None, :]
        additive_mask = (
            1.0 - ops.cast(attention_mask, hidden_states.dtype)
        ) * -1e9
        # Apply bidirectional sliding window for sliding_attention layers.
        if (
            self.attention_type == "sliding_attention"
            and self.sliding_window is not None
        ):
            seq_len = ops.shape(hidden_states)[1]
            # Build position indices for the sliding window mask.
            q_idx = ops.arange(seq_len)[:, None]  # (S, 1)
            kv_idx = ops.arange(seq_len)[None, :]  # (1, S)
            dist = q_idx - kv_idx
            # HF bidirectional window:
            #   left_window  = (sliding_window + 1) // 2
            #   right_window = sliding_window // 2 + 1
            left_w = (self.sliding_window + 1) // 2
            right_w = self.sliding_window // 2 + 1
            window_mask = ((dist >= 0) & (dist < left_w)) | (
                (dist < 0) & (-dist < right_w)
            )
            # Expand to (1, 1, S, S) and convert to additive mask.
            window_mask = ops.cast(
                window_mask[None, None, :, :], hidden_states.dtype
            )
            window_additive = (1.0 - window_mask) * -1e9
            additive_mask = additive_mask + window_additive
        return additive_mask

    def call(
        self,
        hidden_states,
        padding_mask=None,
        training=None,
    ):
        residual = hidden_states
        attention_mask = self._make_attention_mask(hidden_states, padding_mask)
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            inputs=hidden_states,
            attention_mask=attention_mask,
            training=training,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + self.dropout(
            hidden_states, training=training
        )
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + self.dropout(
            hidden_states, training=training
        )
        return hidden_states

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "rms_norm_eps": self.rms_norm_eps,
                "head_dim": self.head_dim,
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
                "attn_logit_softcapping": self.attn_logit_softcapping,
                "use_query_key_norm": self.use_query_key_norm,
            }
        )
        return config
