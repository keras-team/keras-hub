import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gpt_oss.gpt_oss_attention import CachedGptOssAttention
from keras_hub.src.models.gpt_oss.gpt_oss_layer_norm import (
    GptOssLayerNormalization,
)
from keras_hub.src.utils.keras_utils import clone_initializer


class GptOssExperts(keras.layers.Layer):
    """Batched feed-forward experts for GPT-OSS (pure keras.ops).

    This layer implements the expert network for the Mixture-of-Experts (MoE)
    block in GPT-OSS. It computes the output for all experts and then
    applies the routing weights to combine their contributions.

    Args:
        num_experts: Integer, total number of experts.
        hidden_dim: Integer, the hidden dimension of the model.
        intermediate_dim: Integer, the intermediate dimension of the expert.
        alpha: Float, scaling factor for the GLU activation.
        limit: Float, clamping limit for gate and up projections.
        kernel_initializer: Initializer for the dense layer kernels.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(
        self,
        num_experts,
        hidden_dim,
        intermediate_dim,
        alpha=1.702,
        limit=7.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = alpha
        self.limit = limit
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, _):
        self._expert_feedforward_gate_up_proj = self.add_weight(
            shape=(
                self.num_experts,
                self.hidden_dim,
                2 * self.intermediate_dim,
            ),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.variable_dtype,
            name="expert_feedforward_gate_up_proj",
        )
        # Bias for gate_up_proj: [num_experts, 2 * intermediate_dim]
        self._expert_feedforward_gate_up_proj_bias = self.add_weight(
            shape=(self.num_experts, 2 * self.intermediate_dim),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
            name="expert_feedforward_gate_up_proj_bias",
        )
        # Weight for down_proj: [num_experts, intermediate_dim, hidden_dim]
        self._expert_feedforward_down_proj = self.add_weight(
            shape=(self.num_experts, self.intermediate_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.variable_dtype,
            name="expert_feedforward_down_proj",
        )
        # Bias for down_proj: [num_experts, hidden_dim]
        self._expert_feedforward_down_proj_bias = self.add_weight(
            shape=(self.num_experts, self.hidden_dim),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
            name="expert_feedforward_down_proj_bias",
        )
        self.built = True

    def call(self, hidden_states, routing_weights):
        # hidden_states: (num_tokens, hidden_dim)
        # routing_weights: (num_tokens, num_experts)

        # Compute gate_up for all experts:
        # (num_tokens, hidden_dim)
        # -> (num_experts, num_tokens, 2*intermediate_dim)
        gate_up = ops.einsum(
            "th,ehm->etm", hidden_states, self._expert_feedforward_gate_up_proj
        )
        gate_up = (
            gate_up + self._expert_feedforward_gate_up_proj_bias[:, None, :]
        )

        # Split into gate and up
        gate = gate_up[..., ::2]  # (num_experts, num_tokens, intermediate_dim)
        up = gate_up[..., 1::2]  # (num_experts, num_tokens, intermediate_dim)

        # Apply clamping
        gate = ops.clip(gate, min_value=None, max_value=self.limit)
        up = ops.clip(up, min_value=-self.limit, max_value=self.limit)

        # GLU activation: gate * sigmoid(gate * alpha)
        glu = gate * ops.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu  # Element-wise multiplication

        # Compute final output for all experts:
        # (num_experts, num_tokens, intermediate_dim)
        # -> (num_experts, num_tokens, hidden_dim)
        expert_out = ops.einsum(
            "eti,eih->eth", gated_output, self._expert_feedforward_down_proj
        )
        expert_out = (
            expert_out + self._expert_feedforward_down_proj_bias[:, None, :]
        )

        # Apply routing weights
        # routing_weights: (num_tokens, num_experts)
        # Transpose and expand to (num_experts, num_tokens, 1) for broadcasting
        routing_weights_expanded = ops.expand_dims(
            ops.transpose(routing_weights, (1, 0)), axis=-1
        )
        weighted_out = expert_out * routing_weights_expanded

        # Sum contributions from all experts
        # (num_experts, num_tokens, hidden_dim) -> (num_tokens, hidden_dim)
        expert_contribution = ops.sum(weighted_out, axis=0)
        return expert_contribution


class GptOssTopKRouter(keras.layers.Layer):
    """Top-K router for GPT-OSS Mixture-of-Experts.

    This layer computes router logits, selects the top-k experts,
    applies softmax to their logits, and then scatters these probabilities
    back into a full expert score tensor.

    Args:
        num_experts: Integer, total number of experts.
        top_k: Integer, number of experts to select per token.
        hidden_dim: Integer, the hidden dimension of the model.
        kernel_initializer: Initializer for the dense layer kernels.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(
        self,
        num_experts,
        top_k,
        hidden_dim,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, _):
        # Router weight: [num_experts, hidden_dim]
        self._router_weight = self.add_weight(
            shape=(self.num_experts, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.variable_dtype,
            name="router_weight",
        )
        # Router bias: [num_experts]
        self._router_bias = self.add_weight(
            shape=(self.num_experts,),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
            name="router_bias",
        )
        self.built = True

    def call(self, hidden_states):
        # hidden_states: (num_tokens, hidden_dim)

        # Compute router logits: (num_tokens, num_experts)
        router_logits = (
            ops.einsum("th,eh->te", hidden_states, self._router_weight)
            + self._router_bias
        )

        # Get top-k values and indices
        router_top_value, router_indices = ops.top_k(
            router_logits, k=self.top_k
        )

        # Apply softmax to top-k values
        router_top_value = ops.softmax(router_top_value, axis=-1)

        # Scatter top-k probabilities back to a full expert score tensor
        # one_hot_indices: (num_tokens, top_k, num_experts)
        one_hot_indices = ops.one_hot(
            router_indices, self.num_experts, dtype=router_top_value.dtype
        )
        # router_scores: (num_tokens, num_experts)
        router_scores = ops.sum(
            one_hot_indices * ops.expand_dims(router_top_value, axis=-1), axis=1
        )
        return router_scores, router_indices


class GptOssMLP(keras.layers.Layer):
    """GPT-OSS Mixture-of-Experts (MoE) block.

    This layer combines the router and expert networks to perform
    the MoE computation.

    Args:
        hidden_dim: Integer, the hidden dimension of the model.
        intermediate_dim: Integer, the intermediate dimension of the expert.
        num_experts: Integer, total number of experts.
        top_k: Integer, number of experts to select per token.
        alpha: Float, scaling factor for the GLU activation in experts.
        limit: Float, clamping limit for gate and up projections in experts.
        kernel_initializer: Initializer for the dense layer kernels.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts,
        top_k,
        alpha=1.702,
        limit=7.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.limit = limit
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, decoder_sequence_shape):
        self.router = GptOssTopKRouter(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_dim=self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            name="router",
            dtype=self.dtype_policy,
        )
        self.router.build(decoder_sequence_shape)

        self.experts = GptOssExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            alpha=self.alpha,
            limit=self.limit,
            kernel_initializer=self.kernel_initializer,
            name="experts",
            dtype=self.dtype_policy,
        )
        # The experts layer expects (num_tokens, hidden_dim)
        self.experts.build(decoder_sequence_shape)
        self.built = True

    def call(self, hidden_states):
        batch_size, seq_len, _ = ops.shape(hidden_states)
        hidden_states_flattened = ops.reshape(
            hidden_states, (-1, self.hidden_dim)
        )

        router_scores, router_indices = self.router(hidden_states_flattened)
        routed_out = self.experts(
            hidden_states_flattened, routing_weights=router_scores
        )

        out = ops.reshape(routed_out, (batch_size, seq_len, self.hidden_dim))
        return out, router_scores


class GptOssTransformerDecoder(keras.layers.Layer):
    """A single GPT-OSS transformer decoder layer.

    This layer implements the full decoder block, including self-attention
    with sink tokens and a Mixture-of-Experts (MoE) feed-forward network.

    Args:
        intermediate_dim: Integer,the intermediate dimension of
        the MoE experts.
        num_query_heads: Integer, number of attention heads for queries.
        num_key_value_heads: Integer,number of attention heads for keys
        and values.
        num_experts: Integer, total number of experts in the MoE block.
        top_k: Integer, number of experts to select per token in the MoE block.
        rope_max_wavelength: The maximum wavelength for the rotary embedding.
        rope_scaling_factor: Scaling factor for rotary embeddings.
        layer_norm_epsilon: Float, epsilon for layer normalization.
        kernel_initializer: Initializer for the dense layer kernels.
        sliding_window: The size of the sliding window for attention.
        dropout: Dropout rate for attention probabilities.
        use_bias: Whether to include bias terms in the dense projections.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """

    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        num_experts,
        top_k=2,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        kernel_initializer="glorot_uniform",
        sliding_window=4096,
        dropout=0,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.dropout = dropout
        self.sliding_window = sliding_window
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.use_bias = use_bias

        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        # Input Layer Normalization
        self._input_layernorm = GptOssLayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="input_layernorm",
        )
        self._input_layernorm.build(decoder_sequence_shape)

        # Self attention layer.
        self._self_attention_layer = CachedGptOssAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            sliding_window=self.sliding_window,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            use_bias=self.use_bias,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        # Post-attention Layer Normalization
        self._post_attention_layernorm = GptOssLayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self._post_attention_layernorm.build(decoder_sequence_shape)

        # Mixture-of-Experts MLP block
        self._mlp_block = GptOssMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            kernel_initializer=self.kernel_initializer,
            name="mlp",
            dtype=self.dtype_policy,
        )
        self._mlp_block.build(decoder_sequence_shape)

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

        # Input Layer Normalization
        x = self._input_layernorm(decoder_sequence)

        # Self attention block.
        x = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )

        if self_attention_cache is not None:
            x, self_attention_cache = x

        x = x + residual
        residual = x

        # Post-attention Layer Normalization
        x = self._post_attention_layernorm(x)

        # MoE MLP block
        x, router_scores = self._mlp_block(x)

        decoder_output = x + residual

        output = (decoder_output,)

        if self_attention_cache is not None:
            output += (self_attention_cache,)

        # GPT-OSS PyTorch returns router_scores, not router_logits
        output += (router_scores,)

        return output[0] if len(output) == 1 else output

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

        # The lower triangular attention mask
        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_update_index
        )

        # GPT-OSS uses a banded attention mask if sliding window is not None
        if self.sliding_window is not None:
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
        # The output shape is the same as the input shape for the main output.
        # If cache is returned, it's a tuple.
        # If router_scores are returned, it's also a tuple.
        # The actual output shape depends on what is returned.
        # For simplicity, we return the shape of the main output.
        return decoder_sequence_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "sliding_window": self.sliding_window,
                "dropout": self.dropout,
                "use_bias": self.use_bias,
            }
        )
        return config
