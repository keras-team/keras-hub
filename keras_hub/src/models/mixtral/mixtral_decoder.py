import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.mixtral.mixtral_attention import (
    CachedMixtralAttention,
)
from keras_hub.src.models.mixtral.mixtral_layer_norm import (
    MixtralLayerNormalization,
)
from keras_hub.src.utils.keras_utils import clone_initializer



class MixtralMoeExperts(keras.layers.Layer):
    """Batched feed-forward experts for Mixtral (pure keras.ops)."""

    def __init__(
        self,
        num_experts,
        hidden_dim,
        intermediate_dim,
        activation_fn="silu",
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = keras.activations.get(activation_fn)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, _):
        # Weight for gate dense layer: [num_experts, hidden_dim, intermediate_dim]
        self._expert_feedforward_gate_dense = self.add_weight(
            shape=(self.num_experts, self.hidden_dim, self.intermediate_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="expert_feedforward_gate_dense",
        )
        # Weight for intermediate dense layer: [num_experts, hidden_dim, intermediate_dim]
        self._expert_feedforward_intermediate_dense = self.add_weight(
            shape=(self.num_experts, self.hidden_dim, self.intermediate_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="expert_feedforward_intermediate_dense",
        )
        # Weight for output dense layer: [num_experts, intermediate_dim, hidden_dim]
        self._expert_feedforward_output_dense = self.add_weight(
            shape=(self.num_experts, self.intermediate_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="expert_feedforward_output_dense",
        )
        self.built = True

    def call(self, hidden_states):
        # Compute gate output for all experts: [num_experts, tokens, intermediate_dim]
        gate = ops.einsum(
            "th,ehm->etm", hidden_states, self._expert_feedforward_gate_dense
        )
        gate = ops.cast(gate, "float32")  # Match PyTorch SiLU precision
        gate = self.activation(gate)
        gate = ops.cast(gate, self.compute_dtype)

        # Compute intermediate output for all experts: [num_experts, tokens, intermediate_dim]
        intermediate = ops.einsum(
            "th,ehm->etm", hidden_states, self._expert_feedforward_intermediate_dense
        )
        hidden = intermediate * gate  # Element-wise multiplication

        # Compute final output: [num_experts, tokens, hidden_dim]
        out = ops.einsum(
            "eti,eih->eth", hidden, self._expert_feedforward_output_dense
        )
        return out
    
class MixtralSparseMoeBlock(keras.layers.Layer):
    """Mixtral sparse MoE block rewritten in batched style."""

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts,
        top_k,
        router_jitter_noise,
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, decoder_sequence_shape):
        # Router dense layer to compute logits for expert selection
        self._sparse_feedforward_gate_dense = keras.layers.Dense(
            self.num_experts,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            dtype=self.dtype_policy,
            name="sparse_feedforward_gate_dense",
        )
        self._sparse_feedforward_gate_dense.build(decoder_sequence_shape)

        # Batched expert bank
        self.expert_bank = MixtralMoeExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            name="experts",
        )
        self.expert_bank.build(decoder_sequence_shape)
        self.built = True

    def call(self, hidden_states, training=False):
        batch_size, seq_len, _ = ops.shape(hidden_states)
        hidden_states_flattened = ops.reshape(hidden_states, (-1, self.hidden_dim))

        # Apply jitter noise during training if specified
        if training and self.router_jitter_noise > 0:
            random_factors = ops.random.uniform(
                shape=ops.shape(hidden_states_flattened),
                minval=1.0 - self.router_jitter_noise,
                maxval=1.0 + self.router_jitter_noise,
                dtype=hidden_states_flattened.dtype,
            )
            hidden_states_flattened = hidden_states_flattened * random_factors

        # Compute router logits and probabilities
        router_logits = self._sparse_feedforward_gate_dense(hidden_states_flattened)
        router_probs = ops.softmax(router_logits, axis=-1)

        # Select top-k experts and their probabilities
        top_p, top_i = ops.top_k(router_probs, k=self.top_k)
        sum_topk = ops.sum(top_p, axis=-1, keepdims=True)
        top_p = top_p / sum_topk  # Normalize top-k probabilities

        # Create routing weights for all experts
        one_hot = ops.one_hot(top_i, self.num_experts)  # [tokens, top_k, num_experts]
        routing_full = ops.sum(one_hot * top_p[..., None], axis=1)  # [tokens, num_experts]
        routing_full = ops.transpose(routing_full, (1, 0))  # [num_experts, tokens]
        routing_full = ops.cast(routing_full, hidden_states_flattened.dtype)

        # Compute expert outputs in a batched manner
        expert_out = self.expert_bank(hidden_states_flattened)  # [num_experts, tokens, hidden_dim]

        # Weight expert outputs by routing probabilities
        weighted_out = expert_out * routing_full[:, :, None]  # [num_experts, tokens, hidden_dim]
        expert_contribution = ops.sum(weighted_out, axis=0)  # [tokens, hidden_dim]

        # Reshape back to original dimensions
        out = ops.reshape(expert_contribution, (batch_size, seq_len, self.hidden_dim))

        return out, router_logits

class MixtralTransformerDecoder(keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        num_experts,
        top_k,
        router_jitter_noise,
        output_router_logits,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        activation="silu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        sliding_window=512,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise

        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor

        self.dropout = dropout

        self.sliding_window = sliding_window
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.output_router_logits = output_router_logits

        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        # Self attention layer.
        self._self_attention_layer = CachedMixtralAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            sliding_window=self.sliding_window,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._self_attention_layernorm = MixtralLayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layernorm",
        )
        self._self_attention_layernorm.build(decoder_sequence_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        self._sparse_moe_block = MixtralSparseMoeBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            router_jitter_noise=self.router_jitter_noise,
        )
        self._sparse_moe_block.build(decoder_sequence_shape)

        self._feedforward_layernorm = MixtralLayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layernorm",
        )
        self._feedforward_layernorm.build(decoder_sequence_shape)

        self.built = True

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        training=None,
    ):
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )
        residual = decoder_sequence

        x = self._self_attention_layernorm(decoder_sequence)

        # Self attention block.
        x = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
        )

        if self_attention_cache is not None:
            x, self_attention_cache = x

        x = self._self_attention_dropout(x, training=training)

        x = x + residual
        residual = x

        x = self._feedforward_layernorm(x)
        x, router_logits = self._sparse_moe_block(x)

        decoder_output = x + residual

        output = (decoder_output,)

        if self_attention_cache is not None:
            output += (self_attention_cache,)

        if self.output_router_logits:
            output += (router_logits,)

        return output

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        # We need to handle a rectangular causal mask when doing cached
        # decoding. For generative inference, `decoder_sequence` will
        # generally be length 1, and `cache` will be the full generation length.
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]

        cache_update_index = (
            0
            if self_attention_cache_update_index is None
            else self_attention_cache_update_index
        )

        # The lower traingular attention mask
        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_update_index
        )

        # Mistral uses a banded attention mask if sliding window is not None
        if self.sliding_window is not None:
            # Below is a workaround for `ops.triu` for Keras 2.
            # TODO(tirthasheshpatel): Use `ops.triu` once Keras 2 support is
            # removed.
            # causal_mask = ops.triu(causal_mask, k=-self.sliding_window)
            i = ops.arange(output_length)[:, None] + cache_update_index
            j = ops.arange(input_length)[None, :]
            causal_mask_upper = ops.cast(i < j + self.sliding_window, "int32")
            causal_mask = ops.minimum(causal_mask, causal_mask_upper)

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "router_jitter_noise": self.router_jitter_noise,
                "sliding_window": self.sliding_window,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
            }
        )
        return config
