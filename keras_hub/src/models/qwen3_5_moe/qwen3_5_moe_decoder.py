import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.qwen3_5.qwen3_5_attention import Qwen3_5Attention
from keras_hub.src.models.qwen3_5.qwen3_5_gated_delta_net import (
    Qwen3_5GatedDeltaNet,
)
from keras_hub.src.models.qwen3_5.qwen3_5_layers import Qwen3_5LayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


def compute_load_balancing_loss(
    router_logits, num_experts, top_k, attention_mask=None
):
    """Compute the load balancing auxiliary loss for a single MoE layer.

    Args:
        router_logits: Tensor of shape (batch_size * seq_len, num_experts).
        num_experts: Integer, total number of experts.
        top_k: Integer, number of experts to select per token.
        attention_mask: Tensor of shape (batch_size, seq_len, seq_len),
            optional mask for padding.

    Returns:
        Scalar tensor representing the auxiliary loss.
    """
    routing_weights = ops.softmax(router_logits, axis=-1)
    _, selected_experts = ops.top_k(routing_weights, k=top_k)
    expert_mask = ops.one_hot(selected_experts, num_experts)

    if attention_mask is not None:
        batch_size, seq_len, _ = ops.shape(attention_mask)
        flat_mask = ops.any(attention_mask, axis=-1)
        flat_mask = ops.reshape(flat_mask, (-1,))
        expert_attention_mask = ops.expand_dims(flat_mask, axis=-1)
        expert_attention_mask = ops.cast(expert_attention_mask, dtype="float32")

        tokens_per_expert = ops.sum(
            expert_mask * expert_attention_mask[:, None, :], axis=0
        ) / ops.maximum(
            ops.sum(expert_attention_mask[:, None, :], axis=0), 1e-9
        )
        router_prob_per_expert = ops.sum(
            routing_weights * expert_attention_mask, axis=0
        ) / ops.maximum(ops.sum(expert_attention_mask, axis=0), 1e-9)
    else:
        tokens_per_expert = ops.mean(expert_mask, axis=0)
        router_prob_per_expert = ops.mean(routing_weights, axis=0)

    tokens_per_expert = ops.mean(tokens_per_expert, axis=0)
    overall_loss = ops.sum(tokens_per_expert * router_prob_per_expert)
    return overall_loss * num_experts


class Qwen3_5MoeMLP(keras.layers.Layer):
    """A SwiGLU feedforward network layer (used as the shared expert).

    Implements: output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        intermediate_dim: int. The size of the intermediate layer.
        hidden_dim: int. The size of the input and output layers.
        kernel_initializer: Initializer for kernel weights.
    """

    def __init__(
        self,
        intermediate_dim,
        hidden_dim,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim
        self.kernel_initializer = kernel_initializer

    def build(self, decoder_sequence_shape):
        self._feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self._feedforward_gate_dense.build(decoder_sequence_shape)

        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)

        self._feedforward_output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )

        self._feedforward_output_dense.build(
            self._feedforward_gate_dense.compute_output_shape(
                decoder_sequence_shape
            )
        )
        self.built = True

    def call(self, x):
        gate_output = self._feedforward_gate_dense(x)
        gate_output = ops.cast(gate_output, "float32")
        gate_output = ops.silu(gate_output)
        gate_output = ops.cast(gate_output, self.compute_dtype)

        x = self._feedforward_intermediate_dense(x)
        return self._feedforward_output_dense(ops.multiply(x, gate_output))


class Qwen3_5MoeExperts(keras.layers.Layer):
    """A layer that contains a bank of feedforward experts for MoE.

    All experts are stored in fused 3D weight tensors and computed
    via batched einsum for efficiency.

    Args:
        num_experts: int. The total number of experts.
        hidden_dim: int. Input/output dimension of each expert.
        intermediate_dim: int. Intermediate dimension of each expert.
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
        self.kernel_initializer = kernel_initializer

    def build(self, _):
        self._expert_feedforward_gate_dense = self.add_weight(
            shape=(
                self.num_experts,
                self.hidden_dim,
                2 * self.intermediate_dim,
            ),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.variable_dtype,
            name="expert_feedforward_gate_dense",
        )

        self._expert_feedforward_output_dense = self.add_weight(
            shape=(self.num_experts, self.intermediate_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.variable_dtype,
            name="expert_feedforward_output_dense",
        )
        self.built = True

    def call(self, hidden_states):
        gate_up = ops.einsum(
            "th,ehm->etm", hidden_states, self._expert_feedforward_gate_dense
        )
        gate, up = ops.split(gate_up, 2, axis=-1)
        gate = ops.cast(gate, "float32")
        gate = ops.silu(gate)
        gate = ops.cast(gate, hidden_states.dtype)
        hidden = up * gate
        out = ops.einsum(
            "eti,eih->eth", hidden, self._expert_feedforward_output_dense
        )
        return out


class Qwen3_5MoeSparseMoeBlock(keras.layers.Layer):
    """A sparse Mixture-of-Experts (MoE) block with shared expert.

    This block routes each token to top-k experts and also adds the
    output of a shared expert (gated by a sigmoid gate). During
    training, a load-balancing auxiliary loss is added.

    Args:
        hidden_dim: int. Input/output dimension.
        moe_intermediate_dim: int. Intermediate dim per expert.
        shared_expert_intermediate_size: int. Intermediate dim for the
            shared expert.
        num_experts: int. Total number of routed experts.
        top_k: int. Number of experts per token.
        kernel_initializer: Initializer for kernel weights.
        router_aux_loss_coefficient: float. Coefficient for auxiliary loss.
    """

    def __init__(
        self,
        hidden_dim,
        moe_intermediate_dim,
        shared_expert_intermediate_size,
        num_experts,
        top_k,
        kernel_initializer="glorot_uniform",
        router_aux_loss_coefficient=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.moe_intermediate_dim = moe_intermediate_dim
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_initializer = kernel_initializer
        self.router_aux_loss_coefficient = router_aux_loss_coefficient

    def build(self, decoder_sequence_shape):
        self._router_gate = keras.layers.Dense(
            self.num_experts,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="router_gate",
            dtype=self.dtype_policy,
        )
        self._router_gate.build(decoder_sequence_shape)

        self.expert_bank = Qwen3_5MoeExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.moe_intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            name="experts",
            dtype=self.dtype_policy,
        )
        self.expert_bank.build(decoder_sequence_shape)

        self.shared_expert = Qwen3_5MoeMLP(
            intermediate_dim=self.shared_expert_intermediate_size,
            hidden_dim=self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            name="shared_expert",
            dtype=self.dtype_policy,
        )
        self.shared_expert.build(decoder_sequence_shape)

        self._shared_expert_gate = keras.layers.Dense(
            1,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="shared_expert_gate",
            dtype=self.dtype_policy,
        )
        self._shared_expert_gate.build(decoder_sequence_shape)

        self.built = True

    def call(self, hidden_states, attention_mask=None, training=None):
        batch_size, seq_len, _ = ops.shape(hidden_states)
        hidden_states_flat = ops.reshape(hidden_states, (-1, self.hidden_dim))

        shared_expert_output = self.shared_expert(hidden_states_flat)
        shared_gate = ops.sigmoid(self._shared_expert_gate(hidden_states_flat))
        shared_expert_output = shared_gate * shared_expert_output

        router_logits = self._router_gate(hidden_states_flat)
        router_probs = ops.softmax(router_logits, axis=-1)
        top_p, top_i = ops.top_k(router_probs, k=self.top_k)
        top_p = top_p / ops.sum(top_p, axis=-1, keepdims=True)

        one_hot = ops.one_hot(top_i, self.num_experts)
        one_hot = ops.cast(one_hot, top_p.dtype)
        routing_full = ops.sum(one_hot * top_p[..., None], axis=1)
        routing_full = ops.transpose(routing_full, (1, 0))
        routing_full = ops.cast(routing_full, hidden_states_flat.dtype)

        expert_out = self.expert_bank(hidden_states_flat)
        weighted_out = expert_out * routing_full[:, :, None]
        expert_contribution = ops.sum(weighted_out, axis=0)

        combined_output = expert_contribution + shared_expert_output
        out = ops.reshape(
            combined_output, (batch_size, seq_len, self.hidden_dim)
        )

        if training:
            aux_loss = compute_load_balancing_loss(
                router_logits=router_logits,
                num_experts=self.num_experts,
                top_k=self.top_k,
                attention_mask=attention_mask,
            )
            self.add_loss(self.router_aux_loss_coefficient * aux_loss)

        return out, router_logits


class Qwen3_5MoeTransformerDecoder(keras.layers.Layer):
    """A Transformer decoder layer for Qwen3.5 MoE.

    Dispatches between full self-attention and linear attention
    (GatedDeltaNet) based on ``layer_type``. The feedforward is always
    a SparseMoeBlock with shared expert.

    Args:
        layer_type: One of ``"full_attention"`` or ``"linear_attention"``.
        num_query_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value attention heads (GQA).
        head_dim: Dimension of each attention head.
        moe_intermediate_dim: Intermediate dimension per expert.
        shared_expert_intermediate_size: Intermediate dim for shared expert.
        num_experts: Total number of routed experts.
        top_k: Number of experts per token.
        partial_rotary_factor: Fraction of head_dim that gets RoPE.
        rope_max_wavelength: Maximum wavelength for rotary embeddings.
        rope_scaling_factor: Scaling factor for rotary embeddings.
        activation: Activation function for the FFN.
        layer_norm_epsilon: Epsilon for layer norms.
        kernel_initializer: Initializer for projection kernels.
        dropout: Dropout rate.
        sliding_window_size: Sliding window size (full_attention only).
        linear_num_key_heads: Number of key heads (linear_attention).
        linear_num_value_heads: Number of value heads (linear_attention).
        linear_key_head_dim: Key head dim (linear_attention).
        linear_value_head_dim: Value head dim (linear_attention).
        linear_conv_kernel_dim: Conv kernel size (linear_attention).
        mrope_section: M-RoPE section sizes.
        router_aux_loss_coefficient: Coefficient for router auxiliary loss.
    """

    def __init__(
        self,
        layer_type,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        moe_intermediate_dim,
        shared_expert_intermediate_size,
        num_experts,
        top_k,
        partial_rotary_factor=0.25,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        activation="silu",
        layer_norm_epsilon=1e-6,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        sliding_window_size=None,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mrope_section=None,
        router_aux_loss_coefficient=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_type = layer_type
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.moe_intermediate_dim = moe_intermediate_dim
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.dropout = dropout
        self.sliding_window_size = sliding_window_size
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.mrope_section = mrope_section
        self.router_aux_loss_coefficient = router_aux_loss_coefficient
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        if self.layer_type == "linear_attention":
            self._linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=self.hidden_dim,
                linear_num_key_heads=self.linear_num_key_heads,
                linear_num_value_heads=self.linear_num_value_heads,
                linear_key_head_dim=self.linear_key_head_dim,
                linear_value_head_dim=self.linear_value_head_dim,
                linear_conv_kernel_dim=self.linear_conv_kernel_dim,
                layer_norm_epsilon=self.layer_norm_epsilon,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dtype=self.dtype_policy,
                name="linear_attn",
            )
            self._linear_attn.build(decoder_sequence_shape)
        elif self.layer_type == "full_attention":
            self._self_attention_layer = Qwen3_5Attention(
                num_query_heads=self.num_query_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                partial_rotary_factor=self.partial_rotary_factor,
                rope_max_wavelength=self.rope_max_wavelength,
                rope_scaling_factor=self.rope_scaling_factor,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dropout=self.dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                sliding_window_size=self.sliding_window_size,
                mrope_section=self.mrope_section,
                dtype=self.dtype_policy,
                name="self_attention",
            )
            self._self_attention_layer.build(decoder_sequence_shape)
        else:
            raise ValueError(
                f"Unknown layer_type '{self.layer_type}'. "
                "Expected 'full_attention' or 'linear_attention'."
            )

        self._input_layernorm = Qwen3_5LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="input_layernorm",
        )
        self._input_layernorm.build(decoder_sequence_shape)

        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        self.mlp = Qwen3_5MoeSparseMoeBlock(
            hidden_dim=self.hidden_dim,
            moe_intermediate_dim=self.moe_intermediate_dim,
            shared_expert_intermediate_size=(
                self.shared_expert_intermediate_size
            ),
            num_experts=self.num_experts,
            top_k=self.top_k,
            kernel_initializer=self.kernel_initializer,
            router_aux_loss_coefficient=self.router_aux_loss_coefficient,
            dtype=self.dtype_policy,
        )
        self.mlp.build(decoder_sequence_shape)

        self._post_attention_layernorm = Qwen3_5LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self._post_attention_layernorm.build(decoder_sequence_shape)

        self.built = True

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        position_ids=None,
        training=None,
    ):
        residual = decoder_sequence
        x = self._input_layernorm(decoder_sequence)

        if self.layer_type == "linear_attention":
            x = self._linear_attn(
                x,
                attention_mask=decoder_padding_mask,
                cache=self_attention_cache,
                cache_update_index=self_attention_cache_update_index,
                training=training,
            )
            if self_attention_cache is not None:
                x, self_attention_cache = x
        elif self.layer_type == "full_attention":
            self_attention_mask = self._compute_self_attention_mask(
                decoder_sequence=decoder_sequence,
                decoder_padding_mask=decoder_padding_mask,
                decoder_attention_mask=decoder_attention_mask,
                self_attention_cache=self_attention_cache,
                self_attention_cache_update_index=(
                    self_attention_cache_update_index
                ),
            )
            x = self._self_attention_layer(
                hidden_states=x,
                attention_mask=self_attention_mask,
                cache=self_attention_cache,
                cache_update_index=self_attention_cache_update_index,
                position_ids=position_ids,
            )
            if self_attention_cache is not None:
                x, self_attention_cache = x

        x = self._self_attention_dropout(x, training=training)
        x = x + residual

        residual = x
        x = self._post_attention_layernorm(x)
        x = self.mlp(x, training=training)

        if isinstance(x, tuple):
            x, _ = x

        x = ops.cast(x, ops.dtype(residual))
        decoder_output = x + residual

        if self_attention_cache is not None:
            if self.layer_type == "linear_attention":
                return (
                    decoder_output,
                    self_attention_cache[0],
                    self_attention_cache[1],
                )
            return decoder_output, self_attention_cache
        return decoder_output

    def call_and_update_cache(
        self,
        decoder_sequence,
        kv_cache,
        conv_cache,
        recurrent_cache,
        cache_update_index,
        decoder_padding_mask=None,
        position_ids=None,
    ):
        """Forward pass with a uniform cache interface.

        Each layer type updates only its own cache and passes the others
        through unchanged.

        Args:
            decoder_sequence: Hidden states (batch, seq_len, hidden_dim).
            kv_cache: KV cache slice for this layer.
            conv_cache: Conv cache slice for this layer.
            recurrent_cache: Recurrent cache slice for this layer.
            cache_update_index: Int, current step index.
            decoder_padding_mask: Optional padding mask.
            position_ids: Optional M-RoPE position IDs.

        Returns:
            Tuple of (output, updated_kv_cache, updated_conv_cache,
            updated_recurrent_cache).
        """
        if self.layer_type == "full_attention":
            output, updated_kv = self(
                decoder_sequence,
                decoder_padding_mask=decoder_padding_mask,
                self_attention_cache=kv_cache,
                self_attention_cache_update_index=cache_update_index,
                position_ids=position_ids,
            )
            return output, updated_kv, conv_cache, recurrent_cache
        else:
            output, updated_conv, updated_recurrent = self(
                decoder_sequence,
                decoder_padding_mask=decoder_padding_mask,
                self_attention_cache=(conv_cache, recurrent_cache),
                self_attention_cache_update_index=cache_update_index,
            )
            return output, kv_cache, updated_conv, updated_recurrent

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence,
            decoder_padding_mask,
            decoder_attention_mask,
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]

        cache_update_index = (
            0
            if self_attention_cache_update_index is None
            else self_attention_cache_update_index
        )
        causal_mask = compute_causal_mask(
            batch_size,
            input_length,
            output_length,
            cache_update_index,
        )
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
                "layer_type": self.layer_type,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "shared_expert_intermediate_size": (
                    self.shared_expert_intermediate_size
                ),
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "sliding_window_size": self.sliding_window_size,
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_num_value_heads": self.linear_num_value_heads,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
                "mrope_section": self.mrope_section,
                "router_aux_loss_coefficient": (
                    self.router_aux_loss_coefficient
                ),
            }
        )
        return config
