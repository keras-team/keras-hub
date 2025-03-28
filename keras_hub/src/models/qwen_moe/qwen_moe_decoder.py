import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.qwen.qwen_attention import QwenAttention
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


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

        self._feedforward_layernorm = QwenLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layernorm",
        )
        self._feedforward_layernorm.build(decoder_sequence_shape)
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


class QwenSparseMoeBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        moe_intermediate_dim,
        shared_expert_intermediate_dim,
        num_experts,
        top_k,
        norm_topk_prob,
        kernel_initializer="glorot_uniform",
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.moe_intermediate_dim = moe_intermediate_dim
        self.shared_expert_intermediate_dim = shared_expert_intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.kernel_initializer = kernel_initializer
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, decoder_sequence_shape):
        self._sparse_feedforward_gate_dense = keras.layers.Dense(
            self.num_experts,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="sparse_feedforward_gate_dense",
        )
        self._sparse_feedforward_gate_dense.build(decoder_sequence_shape)

        self.experts = [
            QwenMoeMLP(
                intermediate_dim=self.moe_intermediate_dim,
                hidden_dim=self.hidden_dim,
                kernel_initializer=self.kernel_initializer,
                layer_norm_epsilon=self.layer_norm_epsilon,
            )
            for _ in range(self.num_experts)
        ]
        for expert in self.experts:
            expert.build(decoder_sequence_shape)

        self.shared_expert_dense = QwenMoeMLP(
            intermediate_dim=self.shared_expert_intermediate_dim,
            hidden_dim=self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            layer_norm_epsilon=self.layer_norm_epsilon,
        )
        self.shared_expert_dense.build(decoder_sequence_shape)

        self.shared_expert_gate_dense = keras.layers.Dense(1, use_bias=False)
        self.shared_expert_gate_dense.build(decoder_sequence_shape)
        self.built = True

    def call(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        router_logits = self._sparse_feedforward_gate_dense(hidden_states)

        routing_weights = ops.softmax(router_logits, axis=1)
        routing_weights, selected_experts = ops.top_k(
            routing_weights, self.top_k
        )

        if self.norm_topk_prob:
            routing_weights /= ops.sum(routing_weights, axis=-1, keepdims=True)

        routing_weights = ops.cast(routing_weights, hidden_states.dtype)

        final_hidden_states = ops.zeros(
            (batch_size * seq_len, hidden_dim), dtype=hidden_states.dtype
        )

        expert_mask = ops.one_hot(
            selected_experts, num_classes=self.num_experts
        )
        expert_mask = ops.transpose(expert_mask, axes=[2, 1, 0])

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = ops.where(expert_mask[expert_idx])
            if ops.shape(top_x)[0] == 0:
                continue  # skip if no tokens routed to this expert

            # Gather relevant hidden states and compute expert output
            current_state = ops.take(hidden_states, top_x, axis=0)
            expert_output = expert_layer(current_state)

            # Apply routing weights
            current_hidden_states = (
                expert_output * routing_weights[top_x, idx, None]
            )

            # Gather current values at top_x from final_hidden_states
            existing_values = ops.take(final_hidden_states, top_x, axis=0)

            # Accumulate: existing + new (mimic index_add)
            updated_values = existing_values + current_hidden_states

            # Scatter the updated values back
            final_hidden_states = ops.scatter_update(
                final_hidden_states, top_x[:, None], updated_values
            )

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = (
            ops.sigmoid(self.shared_expert_gate_dense(hidden_states))
            * shared_expert_output
        )

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, seq_len, hidden_dim
        )

        return final_hidden_states, router_logits


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
        norm_topk_prob,
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
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.output_router_logits = output_router_logits

        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        # Self attention layer.
        self._self_attention_layer = QwenAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            use_sliding_window_attention=self.use_sliding_window_attention,
            sliding_window_size=self.sliding_window_size,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._self_attention_layernorm = QwenLayerNorm(
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
                norm_topk_prob=self.norm_topk_prob,
                kernel_initializer=self.kernel_initializer,
            )
            self.mlp.build(decoder_sequence_shape)
        else:
            self.mlp = QwenMoeMLP(
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
            )
            self.mlp.build(decoder_sequence_shape)

        self._feedforward_layernorm = QwenLayerNorm(
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
        x = self.mlp(x)
        if isinstance(x, tuple):
            x, router_logits = x
        else:
            router_logits = None

        decoder_output = x + residual

        output = (decoder_output,)

        if self_attention_cache is not None:
            output += self_attention_cache

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
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
            }
        )
        return config
