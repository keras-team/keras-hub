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


class MixtralMoeMLP(keras.layers.Layer):
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


class MixtralSparseMoeBlock(keras.layers.Layer):
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
        self._sparse_feedforward_gate_dense = keras.layers.Dense(
            self.num_experts,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="sparse_feedforward_gate_dense",
        )
        self._sparse_feedforward_gate_dense.build(decoder_sequence_shape)

        self.experts = [
            MixtralMoeMLP(
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
                kernel_initializer=self.kernel_initializer,
                layer_norm_epsilon=self.layer_norm_epsilon,
            )
            for _ in range(self.num_experts)
        ]
        for expert in self.experts:
            expert.build(decoder_sequence_shape)

    def call(self, hidden_states, training=False):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Jitter noise augmentation (training only)
        if training and self.router_jitter_noise > 0:
            random_factors = ops.random.uniform(
                shape=ops.shape(hidden_states),
                minval=1.0 - self.router_jitter_noise,
                maxval=1.0 + self.router_jitter_noise,
                dtype=hidden_states.dtype,
            )
            hidden_states = hidden_states * random_factors

        hidden_states_2d = ops.reshape(hidden_states, (-1, hidden_dim))

        router_logits = self._sparse_feedforward_gate_dense(hidden_states_2d)
        routing_weights = ops.softmax(router_logits, axis=1)

        routing_weights, selected_experts = ops.top_k(
            routing_weights, k=self.top_k
        )
        sum_topk = ops.sum(routing_weights, axis=-1, keepdims=True)
        routing_weights = routing_weights / sum_topk

        routing_weights = ops.cast(routing_weights, hidden_states.dtype)

        # Prepare final hidden states
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
                continue

            # Gather hidden states belonging to this expert
            current_state = ops.take(hidden_states_2d, top_x, axis=0)
            expert_output = expert_layer(current_state)

            # Multiply by routing weights
            # routing_weights is shape (batch_size*seq_len, top_k)
            # We want routing_weights[top_x, idx]
            factor = routing_weights[top_x, idx]
            factor = ops.expand_dims(factor, axis=-1)  # shape = (n_tokens, 1)
            current_hidden_states = expert_output * factor

            existing_values = ops.take(final_hidden_states, top_x, axis=0)
            updated_values = existing_values + current_hidden_states
            final_hidden_states = ops.scatter_update(
                final_hidden_states, top_x[:, None], updated_values
            )

        final_hidden_states = ops.reshape(
            final_hidden_states, (batch_size, seq_len, hidden_dim)
        )

        return final_hidden_states, router_logits


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
