import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_attention import Qwen3OmniMoeAttention
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_layernorm import Qwen3OmniMoeLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


def compute_load_balancing_loss(
    router_logits, num_experts, num_experts_per_tok, attention_mask=None
):
    """
    Compute the load balancing auxiliary loss for a single MoE layer.

    Args:
        router_logits: Tensor of shape (batch_size * seq_len, num_experts).
        num_experts: Integer, total number of experts.
        num_experts_per_tok: Integer, number of experts to select per token.
        attention_mask: Tensor of shape (batch_size, seq_len, seq_len),
            optional mask for padding.

    Returns:
        Scalar tensor representing the auxiliary loss.
    """
    # Compute routing probabilities
    routing_weights = ops.softmax(
        router_logits, axis=-1
    )  # Shape: (batch_size * seq_len, num_experts)

    # Get top-k experts
    _, selected_experts = ops.top_k(
        routing_weights, k=num_experts_per_tok
    )  # Shape: (batch_size * seq_len, num_experts_per_tok)

    # Create one-hot encoding for selected experts
    expert_mask = ops.one_hot(
        selected_experts, num_experts
    )  # Shape: (batch_size * seq_len, num_experts_per_tok, num_experts)

    if attention_mask is not None:
        # Convert attention mask to (batch_size, seq_len)
        batch_size, seq_len, _ = ops.shape(attention_mask)
        flat_mask = ops.any(attention_mask, axis=-1)
        flat_mask = ops.reshape(flat_mask, (-1, 1, 1))  # (batch_size * seq_len, 1, 1)
        expert_mask = expert_mask * flat_mask

    # Compute expert usage
    expert_usage = ops.mean(expert_mask, axis=0)  # Shape: (num_experts,)
    expert_usage = ops.mean(expert_usage, axis=0)  # Shape: (num_experts,)

    # Compute load balancing loss
    num_tokens = ops.sum(ops.any(attention_mask, axis=-1)) if attention_mask is not None else ops.shape(routing_weights)[0]
    expert_usage = expert_usage * num_experts
    load_balancing_loss = ops.sum(expert_usage) * ops.sum(expert_usage) / (num_experts * num_experts)

    return load_balancing_loss


class Qwen3OmniMoeTransformerDecoderLayer(keras.layers.Layer):
    """A transformer decoder layer for Qwen3-Omni MoE model.

    This layer implements a complete transformer decoder layer with self-attention
    and a sparse mixture-of-experts (MoE) feedforward network. It uses pre-normalization
    architecture with RMSNorm for improved training stability.

    Args:
        num_query_heads: int. The number of heads for the query projections.
        num_key_value_heads: int. The number of heads for the key and value
            projections (must be <= num_query_heads).
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the feedforward network.
        num_experts: int. The number of experts in each MoE layer.
        num_experts_per_tok: int. The number of experts to select for each token.
        head_dim: int. The size of each attention head.
        layer_norm_epsilon: float, default 1e-6. The epsilon value used for
            layer normalization.
        dropout: float, default 0.0. Dropout probability.
        sliding_window_size: int, default 4096. Size of the sliding local window.
        max_sequence_length: int, default 32768. The maximum sequence length
            supported by the model.
        dtype: str or `keras.mixed_precision.DTypePolicy`, optional. The dtype
            to use for the layer's computations and weights.

    Example:
    ```python
    # Create decoder layer
    layer = Qwen3OmniMoeTransformerDecoderLayer(
        num_query_heads=32,
        num_key_value_heads=4,
        hidden_dim=4096,
        intermediate_dim=11008,
        num_experts=8,
        num_experts_per_tok=2
    )
    
    # Apply to input
    hidden_states = keras.random.normal((2, 10, 4096))
    outputs = layer(hidden_states)
    # outputs["hidden_states"] shape: (2, 10, 4096)
    # outputs["cache"] contains attention cache
    # outputs["router_logits"] contains MoE routing logits
    ```
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        num_experts,
        num_experts_per_tok,
        head_dim,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        sliding_window_size=4096,
        max_sequence_length=32768,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.head_dim = head_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.sliding_window_size = sliding_window_size
        self.max_sequence_length = max_sequence_length

        # Self-attention
        self.attention = Qwen3OmniMoeAttention(
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_max_wavelength=10000,
            rope_scaling_factor=1.0,
            dropout=dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            sliding_window_size=sliding_window_size,
            dtype=dtype,
            name="attention",
        )

        # MoE feedforward
        self.moe_feedforward = Qwen3OmniMoeSparseMoeBlock(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=True,
            router_aux_loss_coef=0.001,
            dtype=dtype,
            name="moe_feedforward",
        )

        # Layer norms
        self.attention_layer_norm = Qwen3OmniMoeLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="attention_layer_norm",
        )
        self.feedforward_layer_norm = Qwen3OmniMoeLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="feedforward_layer_norm",
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        residual = hidden_states
        hidden_states = self.attention_layer_norm(hidden_states)

        # Self-attention
        attention_result = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )
        
        # Handle return value - can be tensor or (tensor, cache) tuple
        if isinstance(attention_result, tuple):
            attention_output, attention_cache = attention_result
        else:
            attention_output = attention_result
            attention_cache = None

        # Residual connection
        hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.feedforward_layer_norm(hidden_states)

        # MoE feedforward
        feedforward_outputs = self.moe_feedforward(
            hidden_states=hidden_states,
            training=training,
        )
        feedforward_output = feedforward_outputs["hidden_states"]

        # Residual connection
        hidden_states = residual + feedforward_output

        # Collect auxiliary loss
        auxiliary_loss = feedforward_outputs.get("auxiliary_loss")
        if auxiliary_loss is not None:
            self.add_loss(auxiliary_loss)

        # Return just hidden_states for functional model compatibility
        # When cache is needed, the CausalLM will access the layer directly
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "sliding_window_size": self.sliding_window_size,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config


class Qwen3OmniMoeExperts(keras.layers.Layer):
    """Batched MoE experts layer using gate-up-down pattern.

    This layer implements multiple experts as batched weight tensors for
    efficient computation. Uses the SiLU-gated linear unit (GLU) pattern:
    output = down(silu(gate) * up)

    Args:
        num_experts: int. The number of experts.
        hidden_dim: int. The input/output dimension.
        intermediate_dim: int. The intermediate dimension for each expert.
        kernel_initializer: Initializer for kernel weights.
    """

    def __init__(
        self,
        num_experts,
        hidden_dim,
        intermediate_dim,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, input_shape):
        # Match HuggingFace weight shapes exactly for weight loading compatibility
        # gate_up_proj: (num_experts, 2 * intermediate_dim, hidden_dim)
        self.gate_up_proj = self.add_weight(
            name="gate_up_proj",
            shape=(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.variable_dtype,
        )

        # down_proj: (num_experts, hidden_dim, intermediate_dim)
        self.down_proj = self.add_weight(
            name="down_proj",
            shape=(self.num_experts, self.hidden_dim, self.intermediate_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, hidden_states, selected_experts):
        """Forward pass through selected experts.

        Args:
            hidden_states: Input tensor of shape (batch_size * seq_len, hidden_dim).
            selected_experts: Selected expert indices of shape 
                (batch_size * seq_len, num_experts_per_tok).

        Returns:
            Expert outputs of shape (batch_size * seq_len, num_experts_per_tok, hidden_dim).
        """
        # hidden_states: (num_tokens, hidden_dim)
        # selected_experts: (num_tokens, num_experts_per_tok)

        # Gather weights for selected experts
        # gate_up_proj: (num_experts, 2 * intermediate_dim, hidden_dim)
        # After gather: (num_tokens, num_experts_per_tok, 2 * intermediate_dim, hidden_dim)
        selected_gate_up = ops.take(self.gate_up_proj, selected_experts, axis=0)
        
        # down_proj: (num_experts, hidden_dim, intermediate_dim)
        # After gather: (num_tokens, num_experts_per_tok, hidden_dim, intermediate_dim)
        selected_down = ops.take(self.down_proj, selected_experts, axis=0)

        # Gate-up projection: hidden_states @ selected_gate_up.T
        # hidden_states: (t, h), selected_gate_up: (t, e, m, h) where m = 2*intermediate
        # Einsum: sum over h -> result (t, e, m)
        gate_up_output = ops.einsum("th,temh->tem", hidden_states, selected_gate_up)

        # Split into gate and up: each (t, e, intermediate_dim)
        gate, up = ops.split(gate_up_output, 2, axis=-1)

        # Apply SiLU gating: silu(gate) * up
        intermediate = ops.silu(gate) * up

        # Down projection: intermediate @ selected_down.T
        # intermediate: (t, e, m) where m = intermediate_dim
        # selected_down: (t, e, h, m) -> need einsum over m to get (t, e, h)
        expert_outputs = ops.einsum("tem,tehm->teh", intermediate, selected_down)

        return expert_outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_experts": self.num_experts,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config


class Qwen3OmniMoeSparseMoeBlock(keras.layers.Layer):
    """A sparse mixture-of-experts (MoE) block for Qwen3-Omni MoE model.

    This layer implements a sparse MoE feedforward network that routes tokens
    to a subset of experts based on learned routing probabilities. Uses the
    gate_up_proj/down_proj expert pattern with SiLU gating.

    Args:
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The intermediate dimension for each expert.
        num_experts: int. The number of experts in the MoE layer.
        num_experts_per_tok: int. The number of experts to select per token.
        norm_topk_prob: bool. Whether to normalize top-k probabilities.
        router_aux_loss_coef: float. Coefficient for auxiliary load balancing loss.
        kernel_initializer: Initializer for kernel weights.

    Example:
    ```python
    # Create MoE block
    moe_block = Qwen3OmniMoeSparseMoeBlock(
        hidden_dim=4096,
        intermediate_dim=11008,
        num_experts=8,
        num_experts_per_tok=2
    )
    
    # Apply to input
    hidden_states = keras.random.normal((2, 10, 4096))
    outputs = moe_block(hidden_states=hidden_states)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts,
        num_experts_per_tok,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

        # Router (gate)
        self.gate = keras.layers.Dense(
            num_experts,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="gate",
        )

        # Batched experts
        self.experts = Qwen3OmniMoeExperts(
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="experts",
        )

    def call(self, hidden_states, training=None):
        batch_size, seq_len, hidden_dim = ops.shape(hidden_states)

        # Flatten for routing: (batch_size * seq_len, hidden_dim)
        hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))

        # Get router logits: (batch_size * seq_len, num_experts)
        router_logits = self.gate(hidden_states_flat)

        # Compute routing probabilities
        routing_weights = ops.softmax(router_logits, axis=-1)

        # Get top-k experts and their weights
        topk_weights, selected_experts = ops.top_k(
            routing_weights, k=self.num_experts_per_tok
        )

        # Normalize top-k probabilities if enabled
        if self.norm_topk_prob:
            topk_weights = topk_weights / ops.sum(topk_weights, axis=-1, keepdims=True)

        # Get expert outputs: (num_tokens, num_experts_per_tok, hidden_dim)
        expert_outputs = self.experts(hidden_states_flat, selected_experts)

        # Weight and combine expert outputs
        # topk_weights: (num_tokens, num_experts_per_tok)
        # expert_outputs: (num_tokens, num_experts_per_tok, hidden_dim)
        topk_weights_expanded = ops.expand_dims(topk_weights, axis=-1)
        final_output = ops.sum(expert_outputs * topk_weights_expanded, axis=1)

        # Reshape back: (batch_size, seq_len, hidden_dim)
        final_output = ops.reshape(final_output, (batch_size, seq_len, hidden_dim))

        # Compute auxiliary loss for load balancing
        auxiliary_loss = self._compute_load_balancing_loss(router_logits, selected_experts)
        auxiliary_loss = auxiliary_loss * self.router_aux_loss_coef

        return {
            "hidden_states": final_output,
            "router_logits": router_logits,
            "auxiliary_loss": auxiliary_loss,
        }

    def _compute_load_balancing_loss(self, router_logits, selected_experts):
        """Compute auxiliary load balancing loss."""
        # router_logits: (num_tokens, num_experts)
        # selected_experts: (num_tokens, num_experts_per_tok)
        num_tokens = ops.shape(router_logits)[0]

        # Compute routing probabilities
        routing_probs = ops.softmax(router_logits, axis=-1)

        # Mean routing probability per expert
        expert_probs = ops.mean(routing_probs, axis=0)  # (num_experts,)

        # Expert frequency (fraction of tokens routed to each expert)
        expert_mask = ops.one_hot(selected_experts, self.num_experts)
        expert_freq = ops.mean(ops.sum(expert_mask, axis=1), axis=0)  # (num_experts,)

        # Load balancing loss: encourages uniform distribution
        aux_loss = ops.sum(expert_probs * expert_freq) * self.num_experts

        return aux_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "norm_topk_prob": self.norm_topk_prob,
                "router_aux_loss_coef": self.router_aux_loss_coef,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config


@keras_hub_export("keras_hub.models.Qwen3OmniMoeTransformerDecoder")
class Qwen3OmniMoeTransformerDecoder(keras.layers.Layer):
    """A transformer decoder for Qwen3-Omni MoE model.

    This layer implements a stack of transformer decoder layers with sparse
    mixture-of-experts (MoE) feedforward networks. Each layer includes self-attention
    and MoE feedforward with pre-normalization architecture.

    Args:
        num_layers: int. The number of transformer decoder layers.
        num_query_heads: int. The number of heads for the query projections.
        num_key_value_heads: int. The number of heads for the key and value
            projections (must be <= num_query_heads).
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the feedforward network.
        num_experts: int. The number of experts in each MoE layer.
        num_experts_per_tok: int. The number of experts to select for each token.
        head_dim: int. The size of each attention head.
        layer_norm_epsilon: float, default 1e-6. The epsilon value used for
            layer normalization.
        dropout: float, default 0.0. Dropout probability.
        sliding_window_size: int, default 4096. Size of the sliding local window.
        max_sequence_length: int, default 32768. The maximum sequence length
            supported by the model.
        dtype: str or `keras.mixed_precision.DTypePolicy`, optional. The dtype
            to use for the layer's computations and weights.

    Example:
    ```python
    # Create transformer decoder
    decoder = Qwen3OmniMoeTransformerDecoder(
        num_layers=32,
        num_query_heads=32,
        num_key_value_heads=4,
        hidden_dim=4096,
        intermediate_dim=11008,
        num_experts=8,
        num_experts_per_tok=2
    )
    
    # Apply to input
    hidden_states = keras.random.normal((2, 10, 4096))
    outputs = decoder(hidden_states)
    # outputs["hidden_states"] shape: (2, 10, 4096)
    # outputs["cache"] contains attention caches from all layers
    # outputs["all_router_logits"] contains MoE routing logits from all layers
    ```
    """

    def __init__(
        self,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        num_experts,
        num_experts_per_tok,
        head_dim,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        sliding_window_size=4096,
        max_sequence_length=32768,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.head_dim = head_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.sliding_window_size = sliding_window_size
        self.max_sequence_length = max_sequence_length

        # Transformer layers
        self.layers = []
        for i in range(num_layers):
            layer = Qwen3OmniMoeTransformerDecoderLayer(
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                head_dim=head_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                sliding_window_size=sliding_window_size,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
                name=f"layer_{i}",
            )
            self.layers.append(layer)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        training=None,
    ):
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                training=training,
            )

        return {"hidden_states": hidden_states}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "sliding_window_size": self.sliding_window_size,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
