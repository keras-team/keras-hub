import keras

from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.models.t5gemma.t5gemma_attention import T5GemmaAttention
from keras_hub.src.models.t5gemma.t5gemma_layers import T5GemmaMLP


class T5GemmaEncoderLayer(keras.layers.Layer):
    """Encoder layer for the T5Gemma model.

    This layer implements a single encoder block in the T5Gemma architecture,
    comprising self-attention and a feed-forward network (MLP).

    Args:
        hidden_size: int, The dimensionality of the hidden states.
        rms_norm_eps: float, The epsilon value for RMS normalization.
        num_attention_heads: int, The number of attention heads in
            self-attention.
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
        initializer_range: float, The range for the random normal initializer.
        attention_dropout: float, The dropout rate applied to attention weights.
        layer_type: str, Type of attention layer, e.g., `"sliding_attention"`.
        head_dim: int, The dimensionality of each attention head.
        attn_logit_softcapping: float, optional, The softcapping value for
            attention logits. Defaults to `None`.
        sliding_window: int, optional, The window size for sliding attention.
            Required if `layer_type` is `"sliding_attention"`. Defaults to
            `None`.
        rope_max_wavelength: float, The maximum wavelength for Rotary Positional
            Embeddings. Defaults to `10000.0`.
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
        initializer_range,
        attention_dropout,
        layer_type,
        head_dim,
        attn_logit_softcapping=None,
        sliding_window=None,
        rope_max_wavelength=10000.0,
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
        self.layer_type = layer_type
        self.sliding_window = sliding_window
        self.rope_max_wavelength = rope_max_wavelength
        self.head_dim = head_dim
        self.attn_logit_softcapping = attn_logit_softcapping
        if (
            self.layer_type == "sliding_attention"
            and self.sliding_window is None
        ):
            raise ValueError(
                "`sliding_window` must be set for `sliding_attention` layer "
                "type."
            )
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
            name="pre_self_attention_layernorm",
        )
        self.post_self_attn_layernorm = RMSNormalization(
            epsilon=rms_norm_eps,
            dtype=self.dtype_policy,
            name="post_self_attention_layernorm",
        )

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
            1.0 - keras.ops.cast(attention_mask, hidden_states.dtype)
        ) * -1e9
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
        # Isometric.
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
                "layer_type": self.layer_type,
                "sliding_window": self.sliding_window,
                "rope_max_wavelength": self.rope_max_wavelength,
                "attn_logit_softcapping": self.attn_logit_softcapping,
            }
        )
        return config
