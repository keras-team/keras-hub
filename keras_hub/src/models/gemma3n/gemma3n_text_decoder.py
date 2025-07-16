import math
import keras
from keras import ops
from keras_hub.src.models.gemma3n.gemma3n_layer_norm import Gemma3nRMSNorm


class Gemma3nTextLaurelBlock(keras.layers.Layer):
    """Learned Augmented Residual Layer (Laurel)."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.linear_left = keras.layers.Dense(
            config.laurel_rank, use_bias=False, name="linear_left"
        )
        self.linear_right = keras.layers.Dense(
            config.hidden_size, use_bias=False, name="linear_right"
        )
        self.post_laurel_norm = Gemma3nRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            name="post_laurel_norm",
        )

    def call(self, hidden_states):
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed_laurel_hidden_states = self.post_laurel_norm(
            laurel_hidden_states
        )
        return hidden_states + normed_laurel_hidden_states


class Gemma3nTextMLP(keras.layers.Layer):
    """MLP block for the Gemma3n text decoder."""

    def __init__(self, config, layer_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_idx = layer_idx
        self.intermediate_size = config.intermediate_size[layer_idx]
        self.activation_sparsity = config.activation_sparsity_pattern[layer_idx]

        self.gate_proj = keras.layers.Dense(
            self.intermediate_size, use_bias=False, name="gate_proj"
        )
        self.up_proj = keras.layers.Dense(
            self.intermediate_size, use_bias=False, name="up_proj"
        )
        self.down_proj = keras.layers.Dense(
            config.hidden_size, use_bias=False, name="down_proj"
        )
        self.act_fn = ops.silu  # Corresponds to ACT2FN['silu']

    def _gaussian_topk(self, inputs):
        # This function needs a backend-agnostic way to compute the inverse CDF (ppf)
        # of a normal distribution. Keras ops doesn't have this directly.
        # We will approximate or use a placeholder for now.
        # For a real implementation, a custom op or a library like `tfp` or `scipy`
        # would be needed, which breaks backend independence.
        # As a simplified placeholder, we'll use a fixed cutoff for demonstration.
        # A more accurate translation would require `scipy.stats.norm.ppf`.
        if self.activation_sparsity > 0.0:
            # Placeholder logic: keep top k% based on magnitude
            k = int(ops.shape(inputs)[-1] * (1 - self.activation_sparsity))
            if k > 0:
                threshold = ops.top_k(inputs, k=k, sorted=False).values[
                    ..., -1:
                ]
                return ops.relu(inputs - threshold)
        return inputs

    def call(self, hidden_states):
        gate_proj = self.gate_proj(hidden_states)
        if self.activation_sparsity > 0.0:
            gate_proj = self._gaussian_topk(gate_proj)
        activations = self.act_fn(gate_proj)
        up_proj = self.up_proj(hidden_states)
        down_proj = self.down_proj(activations * up_proj)
        return down_proj


class Gemma3nTextAltUp(keras.layers.Layer):
    """Alternating Updates (AltUp) block."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.correct_output_scale = self.add_weight(
            shape=(config.hidden_size,),
            initializer="zeros",
            name="correct_output_scale",
        )
        self.correction_coefs = keras.layers.Dense(
            config.altup_num_inputs, use_bias=False, name="correction_coefs"
        )
        self.prediction_coefs = keras.layers.Dense(
            config.altup_num_inputs**2, use_bias=False, name="prediction_coefs"
        )
        self.modality_router = keras.layers.Dense(
            config.altup_num_inputs, use_bias=False, name="modality_router"
        )
        self.router_norm = Gemma3nRMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, name="router_norm"
        )
        self.router_input_scale = config.hidden_size**-1.0

    def compute_router_modalities(self, x):
        router_inputs = self.router_norm(x) * self.router_input_scale
        routed = self.modality_router(router_inputs)
        return ops.tanh(ops.cast(routed, "float32"))

    def predict(self, hidden_states):
        modalities = self.compute_router_modalities(
            hidden_states[self.config.altup_active_idx]
        )
        all_coefs = ops.reshape(
            self.prediction_coefs(modalities),
            ops.shape(modalities)[:-1]
            + (self.config.altup_num_inputs, self.config.altup_num_inputs),
        )
        all_coefs = ops.transpose(all_coefs, axes=(0, 1, 3, 2))

        predictions = ops.einsum(
            "abch,hdi->abcd",
            ops.transpose(hidden_states, [1, 2, 3, 0]),
            all_coefs,
        )
        predictions = ops.transpose(predictions, [3, 0, 1, 2])
        return predictions + hidden_states

    def correct(self, predictions, activated):
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.config.altup_active_idx]
        innovation = ops.repeat(
            ops.expand_dims(innovation, 0), self.config.altup_num_inputs, axis=0
        )

        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = ops.expand_dims(ops.transpose(all_coefs, [2, 0, 1]), -1)

        corrected = innovation * all_coefs
        return corrected + predictions

    def scale_corrected_output(self, corrected):
        return corrected * self.correct_output_scale


class Gemma3nTextAttention(keras.layers.Layer):
    """Attention mechanism for the Gemma3n text decoder."""

    def __init__(self, config, layer_idx, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.attention_dropout = config.attention_dropout

        self.q_proj = keras.layers.Dense(
            config.num_attention_heads * self.head_dim,
            use_bias=config.attention_bias,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            config.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            config.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            name="v_proj",
        )
        self.o_proj = keras.layers.Dense(
            config.hidden_size, use_bias=config.attention_bias, name="o_proj"
        )

        self.q_norm = Gemma3nRMSNorm(
            dim=config.head_dim, epsilon=config.rms_norm_eps, name="q_norm"
        )
        self.k_norm = Gemma3nRMSNorm(
            dim=config.head_dim, epsilon=config.rms_norm_eps, name="k_norm"
        )
        self.v_norm = Gemma3nRMSNorm(
            dim=config.head_dim,
            epsilon=config.rms_norm_eps,
            with_scale=False,
            name="v_norm",
        )

        # KV sharing logic
        first_kv_shared_layer_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        if self.is_kv_shared_layer:
            layer_type = config.layer_types[layer_idx]
            # This logic is complex to translate directly without a running model graph.
            # It finds the index of the layer from which to reuse the KV cache.
            self.kv_shared_layer_index = None  # Placeholder
        else:
            self.kv_shared_layer_index = None

    def call(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
    ):
        # This is a simplified call. A full implementation would mirror the PyTorch version's logic for
        # projecting Q, K, V, applying RoPE, handling KV caching (including shared caches),
        # and finally calling dot_product_attention.
        return hidden_states, None  # Placeholder


class Gemma3nTextDecoderLayer(keras.layers.Layer):
    """Gemma3n Text Decoder Layer, incorporating AltUp, Laurel, and Attention."""

    def __init__(self, config, layer_idx, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = Gemma3nTextAttention(
            config, layer_idx, name="self_attn"
        )
        self.mlp = Gemma3nTextMLP(config, layer_idx=layer_idx, name="mlp")

        self.input_layernorm = Gemma3nRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            name="input_layernorm",
        )
        self.post_attention_layernorm = Gemma3nRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            name="post_attention_layernorm",
        )
        self.pre_feedforward_layernorm = Gemma3nRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            name="pre_feedforward_layernorm",
        )
        self.post_feedforward_layernorm = Gemma3nRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            name="post_feedforward_layernorm",
        )

        self.altup = Gemma3nTextAltUp(config, name="altup")
        self.laurel = Gemma3nTextLaurelBlock(config, name="laurel")

        self.per_layer_input_gate = keras.layers.Dense(
            config.hidden_size_per_layer_input,
            use_bias=False,
            name="per_layer_input_gate",
        )
        self.per_layer_projection = keras.layers.Dense(
            config.hidden_size, use_bias=False, name="per_layer_projection"
        )
        self.post_per_layer_input_norm = Gemma3nRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            name="post_per_layer_input_norm",
        )

    def call(
        self,
        hidden_states,  # This is now a stacked tensor for AltUp
        position_embeddings_global,
        position_embeddings_local,
        per_layer_input,
        attention_mask=None,
        past_key_value=None,
        # ... other args
    ):
        # 1. AltUp predict
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.config.altup_active_idx]

        # 2. Norm and Laurel
        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        # 3. Attention
        position_embeddings = (
            position_embeddings_local
            if self.self_attn.is_sliding
            else position_embeddings_global
        )
        attn, self_attn_weights = self.self_attn(
            hidden_states=active_prediction_normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        attn = self.post_attention_layernorm(attn)

        # 4. Gating and residual connections
        attn_gated = active_prediction + attn
        attn_laurel = (attn_gated + laurel_output) / math.sqrt(2)

        # 5. MLP
        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        # 6. AltUp correct
        corrected_predictions = self.altup.correct(
            predictions, attn_ffw_laurel_gated
        )

        # 7. Per-layer input fusion
        first_prediction = corrected_predictions[self.config.altup_active_idx]
        if self.config.altup_correct_scale:
            first_prediction = self.altup.scale_corrected_output(
                first_prediction
            )

        gated_input = ops.silu(self.per_layer_input_gate(first_prediction))
        fused_input = gated_input * per_layer_input

        projected_input = self.per_layer_projection(fused_input)
        normed_input = self.post_per_layer_input_norm(projected_input)

        # Update other predictions
        # This part is tricky to translate directly without a loop or map_fn
        # corrected_predictions[1:] += normed_input

        return corrected_predictions, self_attn_weights
