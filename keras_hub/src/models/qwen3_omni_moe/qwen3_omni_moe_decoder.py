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
    expert_usage = ops.mean(experts_mask, axis=0)  # Shape: (num_experts,)
    expert_usage = ops.mean(expert_usage, axis=0)  # Shape: (num_experts,)

    # Compute load balancing loss
    num_tokens = ops.sum(ops.any(attention_mask, axis=-1)) if attention_mask is not None else ops.shape(routing_weights)[0]
    expert_usage = expert_usage * num_experts
    load_balancing_loss = ops.sum(expert_usage) * ops.sum(expert_usage) / (num_experts * num_experts)

    return load_balancing_loss


class Qwen3OmniMoeTransformerDecoderLayer(keras.layers.Layer):
    """A transformer decoder layer for Qwen3-Omni MoE model."""

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
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            sliding_window_size=sliding_window_size,
            max_sequence_length=max_sequence_length,
            dtype=dtype,
            name="attention",
        )

        # MoE feedforward
        self.moe_feedforward = Qwen3OmniMoeSparseMoeBlock(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
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
        position_ids=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        # Self-attention
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )
        attention_output = attention_outputs["hidden_states"]
        attention_cache = attention_outputs.get("cache")

        # Residual connection
        hidden_states = hidden_states + attention_output

        # MoE feedforward
        feedforward_outputs = self.moe_feedforward(
            hidden_states=hidden_states,
            training=training,
        )
        feedforward_output = feedforward_outputs["hidden_states"]

        # Residual connection
        hidden_states = hidden_states + feedforward_output

        return {
            "hidden_states": hidden_states,
            "cache": attention_cache,
            "router_logits": feedforward_outputs.get("router_logits"),
        }

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


class Qwen3OmniMoeSparseMoeBlock(keras.layers.Layer):
    """A sparse MoE block for Qwen3-Omni MoE model."""

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts,
        num_experts_per_tok,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router
        self.router = keras.layers.Dense(
            num_experts,
            use_bias=False,
            dtype=dtype,
            name="router",
        )

        # Experts
        self.experts = []
        for i in range(num_experts):
            expert = keras.Sequential([
                keras.layers.Dense(
                    intermediate_dim,
                    activation="silu",
                    dtype=dtype,
                    name=f"expert_{i}_up",
                ),
                keras.layers.Dense(
                    hidden_dim,
                    dtype=dtype,
                    name=f"expert_{i}_down",
                ),
            ], name=f"expert_{i}")
            self.experts.append(expert)

    def call(self, hidden_states, training=None):
        batch_size, seq_len, hidden_dim = ops.shape(hidden_states)
        
        # Flatten for routing
        hidden_states_flat = ops.reshape(hidden_states, (-1, hidden_dim))
        
        # Get router logits
        router_logits = self.router(hidden_states_flat)
        
        # Get top-k experts
        routing_weights = ops.softmax(router_logits, axis=-1)
        _, selected_experts = ops.top_k(routing_weights, k=self.num_experts_per_tok)
        
        # Create expert mask
        expert_mask = ops.one_hot(selected_experts, self.num_experts)
        
        # Apply experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states_flat)
            # Get the mask for this expert across all selected experts
            expert_mask_i = expert_mask[:, :, i]  # Shape: (batch_size * seq_len, num_experts_per_tok)
            expert_mask_i = ops.sum(expert_mask_i, axis=1, keepdims=True)  # Sum across selected experts
            expert_output = expert_output * expert_mask_i
            expert_outputs.append(expert_output)
        
        # Combine expert outputs
        final_output = ops.sum(ops.stack(expert_outputs, axis=0), axis=0)
        
        # Reshape back
        final_output = ops.reshape(final_output, (batch_size, seq_len, hidden_dim))
        
        return {
            "hidden_states": final_output,
            "router_logits": router_logits,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
            }
        )
        return config


@keras_hub_export("keras_hub.models.Qwen3OmniMoeTransformerDecoder")
class Qwen3OmniMoeTransformerDecoder(keras.layers.Layer):
    """A transformer decoder for Qwen3-Omni MoE model."""

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
        position_ids=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        # Initialize cache if not provided
        if cache is None:
            cache = [None] * self.num_layers

        # Process through layers
        all_hidden_states = []
        all_router_logits = []
        current_cache = []

        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache[i],
                cache_update_index=cache_update_index,
                training=training,
            )
            
            hidden_states = layer_outputs["hidden_states"]
            current_cache.append(layer_outputs.get("cache"))
            
            if "router_logits" in layer_outputs:
                all_router_logits.append(layer_outputs["router_logits"])

        return hidden_states

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
