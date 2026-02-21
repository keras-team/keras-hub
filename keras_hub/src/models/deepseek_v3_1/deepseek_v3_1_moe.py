"""Mixture-of-Experts (MoE) implementation for DeepSeek V3.1."""

import keras
from keras import ops


class DeepSeekV3_1MoE(keras.layers.Layer):
    """Mixture-of-Experts layer for DeepSeek V3.1.

    Implements DeepSeekMoE with:
    - Sigmoid-based affinity scores (as per paper Section 2.1.2, eq. 15)
    - Top-K routing with normalization of selected scores only
    - Shared experts always active
    - Routed experts selected per token

    Note: The auxiliary-loss-free load balancing bias terms (Section 2.1.2)
    are not implemented here as they are a training-time mechanism. For
    inference, standard top-K routing is used.

    Note: The expert loop computes all experts for correctness in a
    framework-agnostic way. In production, sparse dispatch kernels
    (e.g., via custom CUDA) would be used for efficiency.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_routed_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        # Router: produces per-expert affinity logits
        self.router = keras.layers.Dense(
            self.num_routed_experts,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name=f"{self.name}_router" if getattr(self, "name", None) else "router",
        )

        # Shared Expert (SwiGLU) - always active, not routed
        if self.num_shared_experts > 0:
            shared_dim = self.intermediate_dim * self.num_shared_experts
            self.shared_gate_proj = keras.layers.Dense(
                shared_dim,
                activation="silu",
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
                name="shared_gate_proj",
            )
            self.shared_up_proj = keras.layers.Dense(
                shared_dim,
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
                name="shared_up_proj",
            )
            self.shared_down_proj = keras.layers.Dense(
                self.hidden_dim,
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
                name="shared_down_proj",
            )

        # Routed Experts (SwiGLU)
        self.expert_gate_projs = []
        self.expert_up_projs = []
        self.expert_down_projs = []

        for i in range(self.num_routed_experts):
            self.expert_gate_projs.append(
                keras.layers.Dense(
                    self.intermediate_dim,
                    activation="silu",
                    kernel_initializer=self.kernel_initializer,
                    dtype=self.dtype_policy,
                    name=f"expert_gate_proj_{i}",
                )
            )
            self.expert_up_projs.append(
                keras.layers.Dense(
                    self.intermediate_dim,
                    kernel_initializer=self.kernel_initializer,
                    dtype=self.dtype_policy,
                    name=f"expert_up_proj_{i}",
                )
            )
            self.expert_down_projs.append(
                keras.layers.Dense(
                    self.hidden_dim,
                    kernel_initializer=self.kernel_initializer,
                    dtype=self.dtype_policy,
                    name=f"expert_down_proj_{i}",
                )
            )

        super().build(input_shape)

    def call(self, hidden_states, training=False):
        compute_dtype = hidden_states.dtype

        # Router in fp32 for numerical stability
        hidden_states_fp32 = ops.cast(hidden_states, "float32")
        router_logits = self.router(hidden_states_fp32)

        # FIX: Use sigmoid for affinity scores, as per paper Section 2.1.2 eq. 15
        # s_{i,t} = Sigmoid(u_t^T e_i)
        # DeepSeek-V3 explicitly switched from softmax (DeepSeek-V2) to sigmoid.
        affinity_scores = ops.sigmoid(router_logits)  # shape: (..., num_routed_experts)

        # Select top-K experts based on affinity scores
        top_k_scores, top_k_indices = ops.top_k(
            affinity_scores, k=self.num_experts_per_tok
        )

        # FIX: Normalize ONLY the selected top-K scores (not all experts).
        # g_{i,t} = s_{i,t} / sum_{j in TopK} s_{j,t}  (eq. 13)
        # This matches the paper: normalization is applied after top-K selection.
        top_k_weights = top_k_scores / (
            ops.sum(top_k_scores, axis=-1, keepdims=True) + 1e-9
        )
        top_k_weights = ops.cast(top_k_weights, compute_dtype)

        final_output = ops.zeros_like(hidden_states)

        # --- Shared Expert Output (always computed, no routing) ---
        if self.num_shared_experts > 0:
            shared_out = self.shared_down_proj(
                self.shared_gate_proj(hidden_states)
                * self.shared_up_proj(hidden_states)
            )
            final_output = final_output + ops.cast(shared_out, compute_dtype)

        # --- Routed Experts Loop ---
        # Note: This iterates all experts for framework compatibility.
        # The mask ensures non-selected experts contribute zero to the output.
        for i in range(self.num_routed_experts):
            # expert_mask: True where this expert is in the top-K for a token
            expert_mask = ops.equal(top_k_indices, i)  # (..., num_experts_per_tok)
            expert_mask_any = ops.any(expert_mask, axis=-1)  # (...,)

            # Expand mask for broadcasting with hidden_dim
            mask_expanded = ops.expand_dims(
                ops.cast(expert_mask_any, compute_dtype), axis=-1
            )

            # Sum the weight contribution from this expert (handles duplicate
            # expert selection gracefully, though top_k prevents duplicates)
            weight_mask = ops.cast(expert_mask, compute_dtype)
            weight = ops.sum(
                top_k_weights * weight_mask, axis=-1, keepdims=True
            )  # (..., 1)

            # SwiGLU computation for the current routed expert
            expert_out = self.expert_down_projs[i](
                self.expert_gate_projs[i](hidden_states)
                * self.expert_up_projs[i](hidden_states)
            )
            expert_out = ops.cast(expert_out, compute_dtype)

            # Weighted sum: zero contribution where this expert is not selected
            final_output = final_output + (expert_out * weight * mask_expanded)

        return final_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_routed_experts": self.num_routed_experts,
                "num_shared_experts": self.num_shared_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
            }
        )
        return config
