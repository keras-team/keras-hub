import math

import keras
from keras import ops

from keras_hub.src.models.gemma3n.gemma3n_attention import Gemma3nTextAltUp
from keras_hub.src.models.gemma3n.gemma3n_attention import Gemma3nTextAttention
from keras_hub.src.models.gemma3n.gemma3n_layer_norm import Gemma3nRMSNorm


def _polyval_impl(coeffs, x):
    """A backend-agnostic implementation of polynomial evaluation."""
    val = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        val = val * x + c
    return val
# === END: ADD THIS HELPER FUNCTION ===


def _custom_erfinv(y):
    """
    A backend-agnostic numerical approximation of the inverse error function (erfinv).
    Based on a well-known rational approximation.
    """
    # Coefficients for the rational approximation
    a = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
    b = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
    c = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
    d = [3.543889200, 1.637067800]

    y_abs = ops.absolute(y)

    # Central region approximation
    y_le_0_7 = ops.less_equal(y_abs, 0.7)
    z = y * y
    
    # === START: REPLACE ops.polyval WITH _polyval_impl ===
    # Keras 2 -> [c3, c2, c1, c0]; Keras 3 -> [c0, c1, c2, c3]
    num = _polyval_impl(a, z)
    den = _polyval_impl([1.0] + b, z)
    # === END: REPLACE ===
    x_central = y * num / den

    # Tail region approximation
    log_val = ops.log(1.0 - y_abs)
    safe_log_val = ops.where(
        ops.isinf(log_val), ops.cast(0.0, log_val.dtype), log_val
    )
    z = ops.sqrt(-safe_log_val)

    # === START: REPLACE ops.polyval WITH _polyval_impl ===
    num = _polyval_impl(c, z)
    den = _polyval_impl([1.0] + d, z)
    # === END: REPLACE ===
    x_tail = ops.sign(y) * num / den

    return ops.where(y_le_0_7, x_central, x_tail)


def _custom_ppf(p):
    """
    Custom implementation of the Percent Point Function (PPF) or inverse CDF
    for the standard normal distribution. It uses the `_custom_erfinv` helper.

    Formula: ppf(p) = sqrt(2) * erfinv(2p - 1)
    """
    return math.sqrt(2.0) * _custom_erfinv(2.0 * p - 1.0)


class Gemma3nTextLaurelBlock(keras.layers.Layer):
    """[Refactored] Learned Augmented Residual Layer (Laurel)."""

    def __init__(self, hidden_size, laurel_rank, rms_norm_eps, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.laurel_rank = laurel_rank
        self.rms_norm_eps = rms_norm_eps

    def build(self, input_shape):
        self.linear_left = keras.layers.Dense(
            self.laurel_rank, use_bias=False, name="linear_left"
        )
        self.linear_right = keras.layers.Dense(
            self.hidden_size, use_bias=False, name="linear_right"
        )
        self.post_laurel_norm = Gemma3nRMSNorm(
            # self.hidden_size, 
            epsilon=self.rms_norm_eps, 
            name="post_laurel_norm"
        )
        self.built = True

    def call(self, hidden_states):
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(
            laurel_hidden_states
        )
        return hidden_states + normed_laurel_hidden_states


class Gemma3nTextMLP(keras.layers.Layer):
    """[Refactored] MLP block for the Gemma3n text decoder."""

    def __init__(
        self, hidden_size, intermediate_size, activation_sparsity, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_sparsity = activation_sparsity
        self.act_fn = ops.silu

    def build(self, input_shape):
        self.gate_proj = keras.layers.Dense(
            self.intermediate_size, use_bias=False, name="gate_proj"
        )
        self.up_proj = keras.layers.Dense(
            self.intermediate_size, use_bias=False, name="up_proj"
        )
        self.down_proj = keras.layers.Dense(
            self.hidden_size, use_bias=False, name="down_proj"
        )
        self.built = True

    def _gaussian_topk(self, inputs):
        if self.activation_sparsity <= 0.0:
            return inputs

        # ### FIX: Replaced tf.math.ndtri with scipy.stats.norm.ppf ###
        # This is the standard scientific Python equivalent.
        std_multiplier = _custom_ppf(self.activation_sparsity)
        std_multiplier = ops.cast(std_multiplier, inputs.dtype)

        inputs_mean = ops.mean(inputs, axis=-1, keepdims=True)
        inputs_std = ops.std(inputs, axis=-1, keepdims=True)

        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return ops.relu(inputs - cutoff_x)

    def call(self, hidden_states):
        gate_proj = self.gate_proj(hidden_states)
        gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj


class Gemma3nTransformerDecoder(keras.layers.Layer):
    """[Refactored] Gemma3n Text Decoder Layer."""

    def __init__(
        self,
        is_sliding_attention,
        altup_correct_scale,
        hidden_size,
        laurel_rank,
        rms_norm_eps,
        intermediate_size,
        activation_sparsity,
        altup_num_inputs,
        altup_active_idx,
        altup_coef_clip,
        head_dim,
        num_attention_heads,
        num_key_value_heads,
        attention_bias,
        attention_dropout,
        hidden_size_per_layer_input,
        rope_wavelength,
        rope_scaling_factor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_sliding_attention = is_sliding_attention
        self.altup_correct_scale = altup_correct_scale
        self.hidden_size = hidden_size
        self.laurel_rank = laurel_rank
        self.rms_norm_eps = rms_norm_eps
        self.intermediate_size = intermediate_size
        self.activation_sparsity = activation_sparsity
        self.altup_num_inputs = altup_num_inputs
        self.altup_active_idx = altup_active_idx
        self.altup_coef_clip = altup_coef_clip
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.rope_wavelength = rope_wavelength
        self.rope_scaling_factor = rope_scaling_factor

    def build(self, input_shape):
        self.self_attn = Gemma3nTextAttention(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            attention_bias=self.attention_bias,
            rms_norm_eps=self.rms_norm_eps,
            attention_dropout=self.attention_dropout,
            rope_scaling_factor=self.rope_scaling_factor,
            rope_wavelength=self.rope_wavelength,
            name="self_attn",
        )
        self.mlp = Gemma3nTextMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation_sparsity=self.activation_sparsity,
            name="mlp",
        )
        self.input_layernorm = Gemma3nRMSNorm(
            # self.hidden_size, 
            epsilon=self.rms_norm_eps, 
            name="input_layernorm"
        )
        self.post_attention_layernorm = Gemma3nRMSNorm(
            # self.hidden_size,
            epsilon=self.rms_norm_eps,
            name="post_attention_layernorm",
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(
            # self.hidden_size,
            epsilon=self.rms_norm_eps,
            name="pre_feedforward_layernorm",
        )
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            # self.hidden_size,
            epsilon=self.rms_norm_eps,
            name="post_feedforward_layernorm",
        )
        self.altup = Gemma3nTextAltUp(
            hidden_size=self.hidden_size,
            altup_num_inputs=self.altup_num_inputs,
            altup_active_idx=self.altup_active_idx,
            rms_norm_eps=self.rms_norm_eps,
            altup_coef_clip=self.altup_coef_clip,
            name="altup",
        )
        self.laurel = Gemma3nTextLaurelBlock(
            hidden_size=self.hidden_size,
            laurel_rank=self.laurel_rank,
            rms_norm_eps=self.rms_norm_eps,
            name="laurel",
        )
        self.per_layer_input_gate = keras.layers.Dense(
            self.hidden_size_per_layer_input,
            use_bias=False,
            name="per_layer_input_gate",
        )
        self.per_layer_projection = keras.layers.Dense(
            self.hidden_size, use_bias=False, name="per_layer_projection"
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(
            # self.hidden_size,
            epsilon=self.rms_norm_eps,
            name="post_per_layer_input_norm",
        )
        self.built = True

    def call(
        self,
        hidden_states_stack,
        per_layer_input,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        cache_update_index=0,
        training=False,
    ):
        predictions = self.altup.predict(hidden_states_stack, training=training)
        active_prediction = predictions[self.altup_active_idx]
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn_output, past_key_value = self.self_attn(
            hidden_states=active_prediction_normed,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_update_index=cache_update_index,
        )

        attn_output = self.post_attention_layernorm(attn_output)
        attn_gated = active_prediction + attn_output
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)
        mlp_input = self.pre_feedforward_layernorm(attn_laurel)
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        activated_output = attn_laurel + mlp_output

        corrected_predictions = self.altup.correct(
            predictions, activated_output, training=training
        )
        first_prediction = corrected_predictions[self.altup_active_idx]
        if self.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(
                first_prediction
            )

        gated_input = ops.silu(self.per_layer_input_gate(first_prediction))
        fused_input = gated_input * per_layer_input
        projected_input = self.per_layer_projection(fused_input)
        normed_input = self.post_per_layer_input_norm(projected_input)

        next_hidden_states_stack = ops.concatenate(
            [corrected_predictions[1:], ops.expand_dims(normed_input, 0)],
            axis=0
        )

        if use_cache:
            return next_hidden_states_stack, past_key_value
        
        return next_hidden_states_stack