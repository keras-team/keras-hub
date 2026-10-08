import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.qwen_moe.qwen_moe_attention import QwenMoeAttention
from keras_hub.src.models.qwen_moe.qwen_moe_layernorm import QwenMoeLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


def compute_load_balancing_loss(
    router_logits, num_experts, top_k, attention_mask=None
):
    """
    Compute the load balancing auxiliary loss for a single MoE layer.

    Args:
        router_logits: Tensor of shape (batch_size * seq_len, num_experts).
        num_experts: Integer, total number of experts.
        top_k: Integer, number of experts to select per token.
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
        routing_weights, k=top_k
    )  # Shape: (batch_size * seq_len, top_k)

    # Create one-hot encoding for selected experts
    expert_mask = ops.one_hot(
        selected_experts, num_experts
    )  # Shape: (batch_size * seq_len, top_k, num_experts)

    if attention_mask is not None:
        # Convert attention mask to (batch_size, seq_len)
        batch_size, seq_len, _ = ops.shape(attention_mask)
        flat_mask = ops.any(attention_mask, axis=-1)
        flat_mask = ops.reshape(
            flat_mask, (-1,)
        )  # Shape: (batch_size * seq_len,)
        # Expand mask for broadcasting
        expert_attention_mask = ops.expand_dims(
            flat_mask, axis=-1
        )  # Shape: (batch_size * seq_len, 1)
        expert_attention_mask = ops.cast(expert_attention_mask, dtype="float32")

        # Compute masked means
        tokens_per_expert = ops.sum(
            expert_mask * expert_attention_mask[:, None, :], axis=0
        ) / ops.maximum(
            ops.sum(expert_attention_mask[:, None, :], axis=0), 1e-9
        )  # Shape: (top_k, num_experts)
        router_prob_per_expert = ops.sum(
            routing_weights * expert_attention_mask, axis=0
        ) / ops.maximum(
            ops.sum(expert_attention_mask, axis=0), 1e-9
        )  # Shape: (num_experts,)
    else:
        # Unmasked means
        tokens_per_expert = ops.mean(
            expert_mask, axis=0
        )  # Shape: (top_k, num_experts)
        router_prob_per_expert = ops.mean(
            routing_weights, axis=0
        )  # Shape: (num_experts,)

    # Average over top_k dimension if necessary
    tokens_per_expert = ops.mean(
        tokens_per_expert, axis=0
    )  # Shape: (num_experts,)

    # Compute the loss
    overall_loss = ops.sum(tokens_per_expert * router_prob_per_expert)
    return overall_loss * num_experts


class QwenMoeMLP(keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        hidden_dim,
        activation_fn="silu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.kernel_initializer = kernel_initializer
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, decoder_sequence_shape):
        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)

        self._feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self._feedforward_gate_dense.build(decoder_sequence_shape)

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

        self.activation = keras.activations.get(self.activation_fn)
        self.built = True

    def call(self, x):
        gate_output = self._feedforward_gate_dense(x)

        # Note that we run the activation function in full 32-bit
        # precision since this is what `torch.nn.functional.silu`
        # does. Internally, `torch.nn.functional.silu` converts the
        # inputs to float32, computes SiLU, and converts the outputs
        # back to compute dtype.
        # CPU Kernel: https://github.com/pytorch/pytorch/blob/35c493f2cf9b623bfdc7e6b34dc1cb39690a7919/aten/src/ATen/native/cpu/Activation.cpp#L1221-L1235  # noqa: E501
        # CUDA Kernel: https://github.com/pytorch/pytorch/blob/35c493f2cf9b623bfdc7e6b34dc1cb39690a7919/aten/src/ATen/native/cuda/ActivationSiluKernel.cu  # noqa: E501
        gate_output = ops.cast(gate_output, "float32")
        gate_output = self.activation(gate_output)
        gate_output = ops.cast(gate_output, self.compute_dtype)

        x = self._feedforward_intermediate_dense(x)

        x = self._feedforward_output_dense(ops.multiply(x, gate_output))

        return x


class QwenMoeExperts(keras.layers.Layer):
    """Batched Experts Layer"""

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
        hidden = up * self.activation(gate)
        out = ops.einsum(
            "eti,eih->eth", hidden, self._expert_feedforward_output_dense
        )
        return out


class QwenSparseMoeBlock(keras.layers.Layer):
    """Qwen-2 Sparse Moe Block"""

    def __init__(
        self,
        hidden_dim,
        moe_intermediate_dim,
        shared_expert_intermediate_dim,
        num_experts,
        top_k,
        norm_top_k_prob,
        kernel_initializer="glorot_uniform",
        layer_norm_epsilon=1e-5,
        router_aux_loss_coefficient=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = moe_intermediate_dim
        self.intermediate_dim_shared = shared_expert_intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_top_k_prob = norm_top_k_prob
        self.kernel_initializer = kernel_initializer
        self.layer_norm_epsilon = layer_norm_epsilon
        self.router_aux_loss_coefficient = router_aux_loss_coefficient

    def build(self, decoder_sequence_shape):
        self._sparse_feedforward_gate_dense = keras.layers.Dense(
            self.num_experts,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="sparse_feedforward_gate_dense",
            dtype=self.dtype_policy,
        )
        self._sparse_feedforward_gate_dense.build(decoder_sequence_shape)

        # NOTE: Experts are implemented as a single layer to enable efficient
        # batched computation. Implementing each expert individually is
        # currently avoided due to the lack of `ragged_dot` support in the
        # Keras ops API, which would make individual implementations unstable
        # and prone to bugs.
        self.expert_bank = QwenMoeExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            name="experts",
            dtype=self.dtype_policy,
        )
        self.expert_bank.build(decoder_sequence_shape)

        self.shared_expert_dense = QwenMoeMLP(
            intermediate_dim=self.intermediate_dim_shared,
            hidden_dim=self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            layer_norm_epsilon=self.layer_norm_epsilon,
            name="shared_expert_dense",
            dtype=self.dtype_policy,
        )
        self.shared_expert_dense.build(decoder_sequence_shape)

        self.shared_expert_gate_dense = keras.layers.Dense(
            1,
            use_bias=False,
            name="shared_expert_gate_dense",
            dtype=self.dtype_policy,
        )
        self.shared_expert_gate_dense.build(decoder_sequence_shape)

        self.built = True

    def call(self, hidden_states, attention_mask=None, training=None):
        batch_size, seq_len, _ = ops.shape(hidden_states)
        hidden_states_flattened = ops.reshape(
            hidden_states, (-1, self.hidden_dim)
        )

        router_logits = self._sparse_feedforward_gate_dense(
            hidden_states_flattened
        )
        router_probs = ops.softmax(router_logits, axis=-1)

        top_p, top_i = ops.top_k(router_probs, k=self.top_k)
        if self.norm_top_k_prob:
            top_p = top_p / ops.sum(top_p, axis=-1, keepdims=True)

        one_hot = ops.one_hot(top_i, self.num_experts)
        one_hot = ops.cast(one_hot, top_p.dtype)
        routing_full = ops.sum(one_hot * top_p[..., None], axis=1)
        routing_full = ops.transpose(routing_full, (1, 0))
        routing_full = ops.cast(routing_full, hidden_states_flattened.dtype)

        expert_out = self.expert_bank(hidden_states_flattened)

        weighted_out = expert_out * routing_full[:, :, None]
        expert_contribution = ops.sum(weighted_out, axis=0)

        shared_expert_output = self.shared_expert_dense(hidden_states_flattened)
        shared_expert_output *= ops.sigmoid(
            self.shared_expert_gate_dense(hidden_states_flattened)
        )

        out_flat = expert_contribution + shared_expert_output
        out = ops.reshape(out_flat, (batch_size, seq_len, self.hidden_dim))

        # Compute and add auxiliary loss during training
        if training:
            aux_loss = compute_load_balancing_loss(
                router_logits=router_logits,
                num_experts=self.num_experts,
                top_k=self.top_k,
                attention_mask=attention_mask,
            )
            self.add_loss(self.router_aux_loss_coefficient * aux_loss)

        return out, router_logits


class QwenMoeTransformerDecoder(keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        moe_intermediate_dim,
        shared_expert_intermediate_dim,
        num_experts,
        top_k,
        norm_top_k_prob,
        decoder_sparse_step,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        activation="silu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        dropout=0,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        layer_index=0,
        mlp_only_layers=[],
        output_router_logits=False,
        router_aux_loss_coefficient=0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.dropout = dropout
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.layer_index = layer_index
        self.mlp_only_layers = mlp_only_layers
        self.moe_intermediate_dim = moe_intermediate_dim
        self.shared_expert_intermediate_dim = shared_expert_intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_top_k_prob = norm_top_k_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coefficient = router_aux_loss_coefficient
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        # Self attention layer.
        self._self_attention_layer = QwenMoeAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            use_sliding_window_attention=self.use_sliding_window_attention,
            sliding_window_size=self.sliding_window_size,
            name="self_attention",
            dtype=self.dtype_policy,
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._self_attention_layernorm = QwenMoeLayerNorm(
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

        # Feedforward layers.
        if (self.layer_index not in self.mlp_only_layers) and (
            self.num_experts > 0
            and (self.layer_index + 1) % self.decoder_sparse_step == 0
        ):
            self.mlp = QwenSparseMoeBlock(
                hidden_dim=self.hidden_dim,
                moe_intermediate_dim=self.moe_intermediate_dim,
                shared_expert_intermediate_dim=self.shared_expert_intermediate_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                norm_top_k_prob=self.norm_top_k_prob,
                router_aux_loss_coefficient=self.router_aux_loss_coefficient,
                kernel_initializer=self.kernel_initializer,
                dtype=self.dtype_policy,
            )
            self.mlp.build(decoder_sequence_shape)
        else:
            self.mlp = QwenMoeMLP(
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
                dtype=self.dtype_policy,
            )
            self.mlp.build(decoder_sequence_shape)

        self._feedforward_layernorm = QwenMoeLayerNorm(
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
        """Forward pass for the decoder layer.

        Args:
            decoder_sequence: Input tensor of shape [batch_size, seq_length,
                hidden_size].
            decoder_padding_mask: Mask tensor for padding tokens.
            decoder_attention_mask: Additional attention mask.
            self_attention_cache: Optional cached key and value tensors for
                self-attention.
            self_attention_cache_update_index: Index at which to update the
                cache.
            training: Boolean indicating whether in training mode.

        Returns:
            decoder_output: Output tensor after applying transformer decoder
                block.
            self_attention_cache: Updated cache tensors (if cache is provided).
        """
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
        if isinstance(self.mlp, QwenSparseMoeBlock):
            x = self.mlp(
                x, training=training, attention_mask=self_attention_mask
            )
        else:
            x = self.mlp(x)
        if isinstance(x, tuple):
            x, router_logits = x
        else:
            router_logits = None

        x = ops.cast(x, ops.dtype(residual))
        decoder_output = x + residual

        output = (decoder_output,)

        if self_attention_cache is not None:
            output += (self_attention_cache,)

        if self.output_router_logits:
            output += (router_logits,)

        return output[0] if len(output) == 1 else output

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        """Computes the self-attention mask combining causal, padding and
        attention masks.

        Args:
            decoder_sequence: Input tensor.
            decoder_padding_mask: Mask tensor for padding tokens.
            decoder_attention_mask: Additional attention mask.
            self_attention_cache: Optional cached key and value tensors.
            self_attention_cache_update_index: Index at which to update the
                cache.

        Returns:
            Combined attention mask tensor.
        """
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

        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_update_index
        )

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def compute_output_shape(self, decoder_sequence_shape):
        """Computes the output shape of the layer.

        Args:
            decoder_sequence_shape: Shape of the decoder sequence input.

        Returns:
            Output shape, which is the same as the input shape.
        """
        return decoder_sequence_shape

    def get_config(self):
        """Returns the config of the layer.

        Returns:
            Dictionary containing the parameters used to initialize this layer.
        """
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "intermediate_dim": self.intermediate_dim,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "shared_expert_intermediate_dim": (
                    self.shared_expert_intermediate_dim
                ),
                "rope_max_wavelength": self.rope_max_wavelength,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "norm_top_k_prob": self.norm_top_k_prob,
                "decoder_sparse_step": self.decoder_sparse_step,
                "mlp_only_layers": self.mlp_only_layers,
                "output_router_logits": self.output_router_logits,
                "router_aux_loss_coefficient": self.router_aux_loss_coefficient,
            }
        )
        return config
