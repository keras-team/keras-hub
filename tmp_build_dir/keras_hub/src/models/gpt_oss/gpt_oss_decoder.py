import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gpt_oss.gpt_oss_attention import GptOssAttention
from keras_hub.src.models.gpt_oss.gpt_oss_layer_norm import (
    GptOssLayerNormalization,
)
from keras_hub.src.utils.keras_utils import clone_initializer


class GptOssExperts(keras.layers.Layer):
    """A layer containing the feed-forward expert networks for GPT-OSS.

    This layer implements the expert networks as described in the GPT-OSS
    paper. It uses a custom GLU activation.

    Args:
        num_experts: int. The total number of experts.
        hidden_dim: int. The hidden size of the model.
        intermediate_dim: int. The intermediate size of the feed-forward
            network.
        kernel_initializer: string. The initializer for the kernel
            weights. Defaults to "glorot_uniform".
        alpha: float. The alpha parameter for the custom GLU
            activation. Defaults to `1.702`.
        limit: float. The clamping limit for gate and up
            projections. Defaults to `7.0`.
    """

    def __init__(
        self,
        num_experts,
        hidden_dim,
        intermediate_dim,
        kernel_initializer="glorot_uniform",
        alpha=1.702,
        limit=7.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.alpha = alpha
        self.limit = limit

    def build(self, _):
        self.gate_up_proj = self.add_weight(
            shape=(
                self.num_experts,
                self.hidden_dim,
                2 * self.intermediate_dim,
            ),
            initializer=self.kernel_initializer,
            name="gate_up_proj",
        )
        self.gate_up_proj_bias = self.add_weight(
            shape=(self.num_experts, 2 * self.intermediate_dim),
            initializer="zeros",
            name="gate_up_proj_bias",
        )
        self.down_proj = self.add_weight(
            shape=(self.num_experts, self.intermediate_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            name="down_proj",
        )
        self.down_proj_bias = self.add_weight(
            shape=(self.num_experts, self.hidden_dim),
            initializer="zeros",
            name="down_proj_bias",
        )
        self.built = True

    def call(self, hidden_states):
        # hidden_states shape: (num_tokens, hidden_dim)
        # Einsum for batched matrix multiplication across experts.
        # [num_experts, num_tokens, 2 * intermediate_dim]
        gate_up = ops.einsum("th,ehm->etm", hidden_states, self.gate_up_proj)
        gate_up = gate_up + self.gate_up_proj_bias[:, None, :]

        gate = gate_up[..., ::2]
        up = gate_up[..., 1::2]

        gate = ops.clip(gate, -1e9, self.limit)
        up = ops.clip(up, -self.limit, self.limit)

        glu = gate * ops.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu

        # [num_experts, num_tokens, hidden_dim]
        out = ops.einsum("etm,emh->eth", gated_output, self.down_proj)
        out = out + self.down_proj_bias[:, None, :]
        return out


class GptOssTopKRouter(keras.layers.Layer):
    """A layer for routing tokens to the top-k experts.

    Args:
        num_experts: int. The total number of experts.
        top_k: int. The number of experts to route each token to.
        kernel_initializer: string. The initializer for the kernel
            weights. Defaults to "glorot_uniform".
    """

    def __init__(
        self,
        num_experts,
        top_k,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, hidden_states_shape):
        self.router_dense = keras.layers.Dense(
            self.num_experts,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="router_dense",
        )
        self.router_dense.build(hidden_states_shape)
        self.built = True

    def call(self, hidden_states):
        # hidden_states shape: (num_tokens, hidden_dim)
        router_logits = self.router_dense(hidden_states)

        routing_weights, selected_experts = ops.top_k(
            router_logits, k=self.top_k
        )
        routing_weights = ops.softmax(routing_weights, axis=-1)

        expert_mask = ops.one_hot(selected_experts, self.num_experts)
        expert_mask = ops.cast(expert_mask, dtype=routing_weights.dtype)

        # Shape: (num_tokens, top_k, num_experts)
        weighted_mask = expert_mask * ops.expand_dims(routing_weights, axis=-1)

        # Shape: (num_tokens, num_experts)
        router_scores = ops.sum(weighted_mask, axis=1)

        return router_scores


class GptOssSparseMoeBlock(keras.layers.Layer):
    """GPT-OSS sparse Mixture of Experts (MoE) block.

    This block combines a router and a set of expert networks to implement
    the MoE layer.

    Args:
        hidden_dim: int. The hidden size of the model.
        intermediate_dim: int. The intermediate size of the feed-forward
            network.
        num_experts: int. The total number of experts.
        top_k: int. The number of experts to route each token to.
            Defaults to 2.
        kernel_initializer: string. The initializer for the kernel
            weights. Defaults to "glorot_uniform".
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts,
        top_k=2,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_initializer = kernel_initializer

    def build(self, decoder_sequence_shape):
        self.router = GptOssTopKRouter(
            num_experts=self.num_experts,
            top_k=self.top_k,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="router",
        )
        self.router.build(decoder_sequence_shape)

        self.experts = GptOssExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="experts",
        )
        self.experts.build(decoder_sequence_shape)
        self.built = True

    def call(self, hidden_states):
        batch_size, seq_len, _ = ops.shape(hidden_states)
        hidden_states_flattened = ops.reshape(
            hidden_states, (-1, self.hidden_dim)
        )

        router_scores = self.router(hidden_states_flattened)

        expert_outputs = self.experts(hidden_states_flattened)

        # Weight expert outputs by router scores and sum
        # router_scores shape: (num_tokens, num_experts)
        # expert_outputs shape: (num_experts, num_tokens, hidden_dim)
        # Transpose scores for broadcasting: (num_experts, num_tokens)
        router_scores_t = ops.transpose(router_scores)
        # Expand for broadcasting: (num_experts, num_tokens, 1)
        router_scores_expanded = ops.expand_dims(router_scores_t, axis=-1)

        weighted_outputs = expert_outputs * router_scores_expanded
        final_output = ops.sum(weighted_outputs, axis=0)

        final_output = ops.reshape(
            final_output, (batch_size, seq_len, self.hidden_dim)
        )
        return final_output, router_scores


class GptOssTransformerDecoder(keras.layers.Layer):
    """A GPT-OSS transformer decoder layer.

    This layer implements the transformer decoder block from the GPT-OSS
    model, which includes self-attention and a sparse MoE block.

    Args:
        intermediate_dim: int. The intermediate size of the feed-forward
            network.
        num_query_heads: int. The number of query attention heads.
        num_key_value_heads: int. The number of key and value attention
            heads.
        num_experts: int. The total number of experts in the MoE layer.
        top_k: int. The number of experts to route each token to.
            Defaults to 2.
        output_router_logits: bool. If True, the router logits will
            be returned by the layer. Defaults to False.
        rope_max_wavelength: int. The maximum wavelength for the
            rotary position embedding. Defaults to 10000.
        rope_scaling_factor: float. The scaling factor for the
            rotary position embedding. Defaults to 1.0.
        layer_norm_epsilon: float. The epsilon for layer
            normalization. Defaults to 1e-6.
        kernel_initializer: string. The initializer for the kernel
            weights. Defaults to "glorot_uniform".
        sliding_window: int. The size of the sliding window for
            attention. Defaults to 4096.
        dropout: float. The dropout rate. Defaults to 0.
    """

    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        num_experts,
        top_k=2,
        output_router_logits=False,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        kernel_initializer="glorot_uniform",
        sliding_window=4096,
        dropout=0,
        head_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_router_logits = output_router_logits
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.sliding_window = sliding_window
        self.dropout = dropout
        self.head_dim = head_dim
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]

        self.self_attention_layer = GptOssAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            sliding_window=self.sliding_window,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            head_dim=self.head_dim,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.self_attention_layer.build(decoder_sequence_shape)

        self.input_layernorm = GptOssLayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="input_layernorm",
        )
        self.input_layernorm.build(decoder_sequence_shape)

        self.post_attention_layernorm = GptOssLayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self.post_attention_layernorm.build(decoder_sequence_shape)

        self.sparse_moe_block = GptOssSparseMoeBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="sparse_moe_block",
        )
        self.sparse_moe_block.build(decoder_sequence_shape)

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
        x = self.input_layernorm(decoder_sequence)

        x = self.self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
        )

        if self_attention_cache is not None:
            x, self_attention_cache = x

        x = x + residual
        residual = x

        x = self.post_attention_layernorm(x)
        x, router_logits = self.sparse_moe_block(x)

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
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
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
            batch_size, input_length, output_length, cache_update_index
        )

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "output_router_logits": self.output_router_logits,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "sliding_window": self.sliding_window,
                "dropout": self.dropout,
                "head_dim": self.head_dim,
            }
        )
        return config
