import keras
from keras import ops

from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm


class Gemma4MoEBlock(keras.layers.Layer):
    """Batched block of feed-forward experts for Gemma4 MoE.

    All expert weights are stored as 3-D tensors to enable efficient batched
    computation via ``einsum``.  Each expert applies gated GELU:
    ``down_proj(gelu(gate_proj(x)) * up_proj(x))``.  A per-expert learnable
    scale is applied to the ``down_proj`` output before the outputs are
    combined by the router.

    Args:
        num_experts: int.  Total number of experts.
        hidden_dim: int.  Hidden state dimension (input and output).
        expert_intermediate_dim: int.  Per-expert intermediate dimension.
    """

    def __init__(
        self,
        num_experts,
        hidden_dim,
        expert_intermediate_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_intermediate_dim = expert_intermediate_dim

    def build(self, _):
        E = self.num_experts
        H = self.hidden_dim
        I = self.expert_intermediate_dim

        self.gate_proj = self.add_weight(
            name="gate_proj",
            shape=(E, H, I),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.up_proj = self.add_weight(
            name="up_proj",
            shape=(E, H, I),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.down_proj = self.add_weight(
            name="down_proj",
            shape=(E, I, H),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )
        # Learned per-expert output scale (initialised to 1.0).
        self.per_expert_scale = self.add_weight(
            name="per_expert_scale",
            shape=(E,),
            initializer="ones",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        """Compute all expert outputs for every token.

        Args:
            x: Float tensor of shape ``[T, H]`` (flattened tokens).

        Returns:
            Float tensor of shape ``[E, T, H]`` — one output per
            (expert, token) pair.
        """
        dtype = x.dtype
        # gate/up: [T, H] x [E, H, I] -> [E, T, I]
        gate = ops.einsum("th,ehi->eti", x, ops.cast(self.gate_proj, dtype))
        up_out = ops.einsum("th,ehi->eti", x, ops.cast(self.up_proj, dtype))
        # GELU activation in float32 for numerical precision.
        gate = keras.activations.gelu(
            ops.cast(gate, "float32"), approximate=True
        )
        gate = ops.cast(gate, dtype)
        hidden = gate * up_out  # [E, T, I]
        # down: [E, T, I] x [E, I, H] -> [E, T, H]
        out = ops.einsum(
            "eti,eih->eth", hidden, ops.cast(self.down_proj, dtype)
        )
        # Apply per-expert output scale: [E] -> [E, 1, 1]
        scale = ops.cast(self.per_expert_scale, dtype)
        scale = ops.reshape(scale, (self.num_experts, 1, 1))
        return out * scale  # [E, T, H]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "hidden_dim": self.hidden_dim,
                "expert_intermediate_dim": self.expert_intermediate_dim,
            }
        )
        return config


class Gemma4Router(keras.layers.Layer):
    """Gemma4 MoE router.

    Routes each token to the top-k experts by:
    1. RMSNorm (parameter-free, no scale).
    2. Scale by ``1 / sqrt(hidden_dim)``.
    3. Element-wise multiply with a learnable per-dimension weight vector.
    4. Linear projection to ``num_experts`` logits.
    5. Softmax → top-k selection → renormalise.
    6. Build sparse dispatch-weight tensor ``[T, E]`` via one-hot encoding.

    Args:
        num_experts: int.  Total number of experts.
        num_experts_per_token: int.  Number of experts selected per token (k).
        layer_norm_epsilon: float.  Epsilon for the internal RMSNorm.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_token,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.layer_norm_epsilon = layer_norm_epsilon

        # RMSNorm without a learnable scale (reuse Gemma4VNorm).
        self.rms_norm = Gemma4VNorm(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="rms_norm",
        )
        self.proj = keras.layers.Dense(
            num_experts,
            use_bias=False,
            dtype=self.dtype_policy,
            name="proj",
        )

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        # Build flat shape since we'll call on [T, H].
        flat_shape = (None, hidden_dim)
        self.rms_norm.build(flat_shape)
        self.proj.build(flat_shape)
        # Per-dimension learnable scale (initialised to 1.0).
        self.per_dim_scale = self.add_weight(
            name="per_dim_scale",
            shape=(hidden_dim,),
            initializer="ones",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self._scalar_root_size = float(hidden_dim) ** -0.5
        self.built = True

    def call(self, x):
        """Compute dispatch weights for routing.

        Args:
            x: Float tensor of shape ``[B, S, H]`` (raw hidden states,
               before any pre-FFW normalization).

        Returns:
            dispatch_weights: Float tensor of shape ``[T, E]`` where
               ``T = B * S``.  Top-k entries per row are non-zero and sum to 1.
        """
        shape = ops.shape(x)
        x_flat = ops.reshape(x, (-1, shape[-1]))  # [T, H]

        # RMSNorm (no scale) + scalar + per-dim scale.
        normed = self.rms_norm(x_flat)
        normed = normed * ops.cast(self._scalar_root_size, normed.dtype)
        normed = normed * ops.cast(self.per_dim_scale, normed.dtype)

        # Router logits and probabilities.
        router_logits = self.proj(normed)  # [T, E]
        router_probs = ops.cast(
            ops.softmax(ops.cast(router_logits, "float32"), axis=-1),
            x.dtype,
        )  # [T, E]

        # Top-k selection.
        _, top_k_indices = ops.top_k(
            router_probs, k=self.num_experts_per_token
        )  # [T, top_k]
        top_k_probs = ops.take_along_axis(
            router_probs, top_k_indices, axis=-1
        )  # [T, top_k]

        # Renormalise so selected expert probabilities sum to 1.
        denom = ops.maximum(
            ops.sum(top_k_probs, axis=-1, keepdims=True),
            ops.cast(1e-9, top_k_probs.dtype),
        )
        top_k_probs = top_k_probs / denom  # [T, top_k]

        # Build sparse dispatch-weight matrix: [T, E] via one-hot sum.
        one_hot = ops.one_hot(top_k_indices, self.num_experts)  # [T, top_k, E]
        one_hot = ops.cast(one_hot, top_k_probs.dtype)
        dispatch_weights = ops.sum(
            one_hot * ops.expand_dims(top_k_probs, axis=-1), axis=1
        )  # [T, E]
        return dispatch_weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_token": self.num_experts_per_token,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
