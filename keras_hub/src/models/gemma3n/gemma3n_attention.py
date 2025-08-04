import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma3n.gemma3n_layer_norm import Gemma3nRMSNorm


class Gemma3nTextAltUp(keras.layers.Layer):
    """[Refactored] Alternating Updates (AltUp) block."""

    def __init__(
        self,
        hidden_size,
        altup_num_inputs,
        altup_active_idx,
        rms_norm_eps,
        altup_coef_clip=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.altup_num_inputs = altup_num_inputs
        self.altup_active_idx = altup_active_idx
        self.rms_norm_eps = rms_norm_eps
        self.altup_coef_clip = altup_coef_clip
        self.router_input_scale = hidden_size**-1.0

    # def build(self, input_shape):
        self.correct_output_scale = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            name="correct_output_scale",
        )
        self.correction_coefs = keras.layers.Dense(
            self.altup_num_inputs, use_bias=False, name="correction_coefs"
        )
        self.prediction_coefs = keras.layers.Dense(
            self.altup_num_inputs**2, use_bias=False, name="prediction_coefs"
        )
        self.modality_router = keras.layers.Dense(
            self.altup_num_inputs, use_bias=False, name="modality_router"
        )
        self.router_norm = Gemma3nRMSNorm(
            # self.hidden_size, 
            epsilon=self.rms_norm_eps, 
            name="router_norm"
        )
        self.built = True

    def compute_router_modalities(self, x):
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return ops.tanh(ops.cast(routed, "float32"))

    def predict(self, hidden_states, training=False):
        modalities = self.compute_router_modalities(
            hidden_states[self.altup_active_idx]
        )

        if self.altup_coef_clip is not None and training:
            self.prediction_coefs.kernel.assign(
                ops.clip(
                    self.prediction_coefs.kernel,
                    -self.altup_coef_clip,
                    self.altup_coef_clip,
                )
            )

        all_coefs = ops.reshape(
            self.prediction_coefs(modalities),
            ops.shape(modalities)[:-1]
            + (self.altup_num_inputs, self.altup_num_inputs),
        )
        hidden_states_p = ops.transpose(hidden_states, [1, 2, 3, 0])
        predictions_p = ops.matmul(hidden_states_p, all_coefs)
        predictions = ops.transpose(predictions_p, [3, 0, 1, 2])
        return predictions + hidden_states

    def correct(self, predictions, activated, training=False):
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.altup_active_idx]
        innovation = ops.repeat(
            ops.expand_dims(innovation, 0), self.altup_num_inputs, axis=0
        )

        if self.altup_coef_clip is not None and training:
            self.correction_coefs.kernel.assign(
                ops.clip(
                    self.correction_coefs.kernel,
                    -self.altup_coef_clip,
                    self.altup_coef_clip,
                )
            )

        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = ops.expand_dims(all_coefs, -1)
        all_coefs_p = ops.transpose(all_coefs, [2, 0, 1, 3])
        corrected = innovation * all_coefs_p
        return corrected + predictions

    def scale_corrected_output(self, corrected):
        return corrected * self.correct_output_scale


class Gemma3nTextAttention(keras.layers.Layer):
    """[Final] Attention layer refactored to use a RotaryEmbedding layer."""

    def __init__(
        self,
        hidden_size,
        head_dim,
        num_attention_heads,
        num_key_value_heads,
        attention_bias,
        rms_norm_eps,
        attention_dropout,
        rope_wavelength,
        rope_scaling_factor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.rope_wavelength = rope_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # === Create all sub-layers in __init__ ===
        self.q_proj = keras.layers.Dense(
            self.num_attention_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="v_proj",
        )
        self.o_proj = keras.layers.Dense(
            self.hidden_size, use_bias=self.attention_bias, name="o_proj"
        )

        self.q_norm = Gemma3nRMSNorm(
            epsilon=self.rms_norm_eps, name="q_norm"
        )
        self.k_norm = Gemma3nRMSNorm(
            epsilon=self.rms_norm_eps, name="k_norm"
        )
        
        # The specific fix is here: `scale` -> `with_scale`
        self.v_norm = Gemma3nRMSNorm(
            epsilon=self.rms_norm_eps,
            with_scale=False,  # Corrected keyword argument
            name="v_norm",
        )

        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=self.rope_wavelength,
            scaling_factor=self.rope_scaling_factor,
            feature_axis=-1,
            sequence_axis=1,
        )
        self.dropout_layer = keras.layers.Dropout(self.attention_dropout)

    def build(self, input_shape):
        # build() is now only responsible for marking the layer as built.
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        cache_update_index=0,
    ):
        # ... your call logic remains the same ...
        batch_size, seq_len, _ = ops.shape(hidden_states)

        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        q_states = ops.reshape(
            q_states,
            (batch_size, seq_len, self.num_attention_heads, self.head_dim),
        )
        k_states = ops.reshape(
            k_states,
            (batch_size, seq_len, self.num_key_value_heads, self.head_dim),
        )
        v_states = ops.reshape(
            v_states,
            (batch_size, seq_len, self.num_key_value_heads, self.head_dim),
        )

        # Note: RMSNorm is applied per-head on the last dimension (head_dim)
        q_states = self.q_norm(q_states)
        k_states = self.k_norm(k_states)
        v_states = self.v_norm(v_states)

        q_states = self.rotary_embedding(
            q_states, start_index=cache_update_index
        )
        k_states = self.rotary_embedding(
            k_states, start_index=cache_update_index
        )

        if use_cache and past_key_value is not None:
            k_states = past_key_value.update(
                k_states, cache_update_index, "key"
            )
            v_states = past_key_value.update(
                v_states, cache_update_index, "value"
            )

        k_states = ops.repeat(
            k_states, repeats=self.num_key_value_groups, axis=2
        )
        v_states = ops.repeat(
            v_states, repeats=self.num_key_value_groups, axis=2
        )

        q_states = ops.transpose(q_states, [0, 2, 1, 3])
        k_states = ops.transpose(k_states, [0, 2, 1, 3])
        v_states = ops.transpose(v_states, [0, 2, 1, 3])

        attn_weights = ops.matmul(
            q_states, ops.transpose(k_states, [0, 1, 3, 2])
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights += ops.cast(attention_mask, attn_weights.dtype)

        attn_weights = ops.softmax(ops.cast(attn_weights, "float32"), axis=-1)
        attn_weights = ops.cast(attn_weights, hidden_states.dtype)
        attn_weights = self.dropout_layer(attn_weights)

        attn_output = ops.matmul(attn_weights, v_states)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value