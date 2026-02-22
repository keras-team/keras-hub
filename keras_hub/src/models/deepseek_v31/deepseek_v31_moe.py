"""DeepSeek V31 Mixture-of-Experts layer."""

import keras
from keras import ops


class DeepSeekV31MoE(keras.layers.Layer):
    """Mixture-of-Experts (MoE) layer for DeepSeek V31.

    Implements DeepSeekMoE routing as described in Section 2.1.2 of the paper.
    Each token is routed to `num_experts_per_tok` out of `num_routed_experts`
    routed experts, plus `num_shared_experts` always-active shared experts.

    Routing uses sigmoid-based affinity scores (not softmax), and normalization
    is applied only to the selected top-K scores:

        s_i = sigmoid(u^T e_i)
        g_i = s_i / sum_{j in TopK} s_j   for i in TopK

    Note on load balancing: The auxiliary-loss-free bias terms described in
    Section 2.1.2 are a training-time mechanism and are not implemented here.
    Inference uses standard top-K routing without bias correction.

    This implementation vectorizes the routed expert computation using batched
    tensor operations (`ops.einsum`), which avoids graph bloat and ensures
    compatibility with XLA compilation (`jit_compile=True`).

    Args:
        hidden_dim: int. Dimensionality of input and output hidden states.
        intermediate_dim: int. Inner dimensionality of each expert's SwiGLU FFN.
        num_routed_experts: int. Total number of routed experts. Defaults to
            `256`.
        num_shared_experts: int. Number of always-active shared experts.
            Defaults to `1`.
        num_experts_per_tok: int. Number of routed experts activated per token
            (top-K). Defaults to `8`.
        kernel_initializer: string or initializer. Initializer for all Dense
            kernel weights. Defaults to `"glorot_uniform"`.

    Example:

    ```python
    moe = keras_hub.layers.DeepSeekV31MoE(
        hidden_dim=512,
        intermediate_dim=1024,
        num_routed_experts=8,
        num_shared_experts=1,
        num_experts_per_tok=2,
    )
    hidden = keras.random.normal((2, 16, 512))
    output = moe(hidden)  # (2, 16, 512)
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
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
        # Router: maps hidden states to per-expert affinity logits.
        self.router = keras.layers.Dense(
            self.num_routed_experts,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="router",
        )
        self.router.build(input_shape)

        # Shared experts (SwiGLU) — always active, not gated.
        if self.num_shared_experts > 0:
            shared_dim = self.intermediate_dim * self.num_shared_experts
            self.shared_gate_proj = keras.layers.Dense(
                shared_dim,
                activation="silu",
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
                name="shared_gate_proj",
            )
            self.shared_up_proj = keras.layers.Dense(
                shared_dim,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
                name="shared_up_proj",
            )
            self.shared_down_proj = keras.layers.Dense(
                self.hidden_dim,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
                name="shared_down_proj",
            )
            self.shared_gate_proj.build(input_shape)
            self.shared_up_proj.build(input_shape)
            shared_inner_shape = list(input_shape[:-1]) + [shared_dim]
            self.shared_down_proj.build(shared_inner_shape)

        # Routed experts (SwiGLU)
        # Stacked into batched tensors for vectorized XLA-compatible execution.
        self.expert_gate_kernel = self.add_weight(
            shape=(
                self.num_routed_experts,
                self.hidden_dim,
                self.intermediate_dim,
            ),
            initializer=self.kernel_initializer,
            name="expert_gate_kernel",
        )
        self.expert_up_kernel = self.add_weight(
            shape=(
                self.num_routed_experts,
                self.hidden_dim,
                self.intermediate_dim,
            ),
            initializer=self.kernel_initializer,
            name="expert_up_kernel",
        )
        self.expert_down_kernel = self.add_weight(
            shape=(
                self.num_routed_experts,
                self.intermediate_dim,
                self.hidden_dim,
            ),
            initializer=self.kernel_initializer,
            name="expert_down_kernel",
        )

        super().build(input_shape)

    def call(self, hidden_states, training=False):
        compute_dtype = hidden_states.dtype

        # Run router in float32 for numerical stability.
        router_logits = ops.cast(
            self.router(ops.cast(hidden_states, "float32")), "float32"
        )

        # Sigmoid affinity scores (Section 2.1.2, eq. 15).
        # DeepSeek-V3 uses sigmoid instead of softmax used in V2.
        affinity_scores = ops.sigmoid(router_logits)

        # Top-K selection.
        top_k_scores, top_k_indices = ops.top_k(
            affinity_scores, k=self.num_experts_per_tok
        )

        # Normalize only the selected K scores (eq. 13).
        top_k_weights = top_k_scores / (
            ops.sum(top_k_scores, axis=-1, keepdims=True) + 1e-9
        )
        top_k_weights = ops.cast(top_k_weights, compute_dtype)

        output = ops.zeros_like(hidden_states)

        # Shared expert contribution (always active).
        if self.num_shared_experts > 0:
            shared_out = self.shared_down_proj(
                self.shared_gate_proj(hidden_states)
                * self.shared_up_proj(hidden_states)
            )
            output = output + ops.cast(shared_out, compute_dtype)

        # ===================================================================
        # Vectorized Routed Expert Contributions
        # ===================================================================

        # 1. Create a dense routing mask of shape (..., num_routed_experts)
        # one_hot shape: (..., K, E)
        mask = ops.one_hot(top_k_indices, self.num_routed_experts)

        # Multiply by weights: expand top_k_weights to (..., K, 1) to broadcast
        weights_expanded = ops.expand_dims(top_k_weights, axis=-1)
        mask_weighted = mask * ops.cast(weights_expanded, mask.dtype)

        # Sum over K to get the final per-expert routing weights: shape (..., E)
        router_mask = ops.sum(mask_weighted, axis=-2)
        router_mask = ops.cast(router_mask, compute_dtype)

        # 2. Compute Gate and Up projections for all experts simultaneously
        # hidden_states: (..., H)
        # expert_kernels: (E, H, I)
        # einsum naturally broadcasts over the missing E dimension to compute (..., E, I)
        gate_out = ops.einsum(
            "...h,ehi->...ei", hidden_states, self.expert_gate_kernel
        )
        up_out = ops.einsum(
            "...h,ehi->...ei", hidden_states, self.expert_up_kernel
        )

        # 3. Apply SwiGLU activation and the routing mask
        expert_act = ops.silu(gate_out) * up_out

        # Expand router_mask to (..., E, 1) for broadcasting over I
        router_mask_expanded = ops.expand_dims(router_mask, axis=-1)

        # Zero-out inactive experts and scale active ones by their affinity scores
        expert_act_weighted = expert_act * router_mask_expanded

        # 4. Compute Down projection and sum over experts simultaneously
        # expert_act_weighted: (..., E, I)
        # expert_down_kernel: (E, I, H)
        # This einsum performs the matmul and sums over the E dimension in one step
        # Output shape: (..., H)
        routed_out = ops.einsum(
            "...ei,eih->...h", expert_act_weighted, self.expert_down_kernel
        )

        output = output + routed_out

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_routed_experts": self.num_routed_experts,
                "num_shared_experts": self.num_shared_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config
