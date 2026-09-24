import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.qwen3_moe.qwen3_moe_decoder import Qwen3MoeMLP
from keras_hub.src.models.qwen3_moe.qwen3_moe_decoder import Qwen3SparseMoeBlock
from keras_hub.src.models.qwen3_moe.qwen3_moe_layernorm import Qwen3MoeLayerNorm
from keras_hub.src.models.qwen3_omni.qwen3_omni_attention import (
    Qwen3OmniAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer


class Qwen3OmniTransformerDecoder(keras.layers.Layer):
    """Pre-norm Qwen3-Omni decoder block: M-RoPE attention + MoE / dense FFN.

    Args:
        intermediate_dim: int. Dense FFN intermediate size (when MoE disabled).
        num_query_heads, num_key_value_heads: int. GQA head counts.
        moe_intermediate_dim: int. Per-expert intermediate size.
        head_dim: int. Per-head attention dimension.
        num_experts, top_k, norm_top_k_prob: MoE routing parameters.
        mrope_section: 3-tuple. M-RoPE ``(t, h, w)`` split sizes.
        rope_max_wavelength, rope_scaling_factor, rope_attention_scaling:
            RoPE / scaling parameters.
        layer_norm_epsilon: float. RMS-norm epsilon.
        activation: callable. FFN activation (defaults to SiLU).
        kernel_initializer: initializer. Kernel initializer.
        dropout: float. Attention dropout rate.
        sliding_window_size: int or None. Sliding-window attention size.
        router_aux_loss_coefficient: float. MoE load-balancing coefficient.
        is_sparse_mlp: bool. Use sparse MoE block instead of dense FFN.
        dtype: dtype policy.
    """

    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        moe_intermediate_dim,
        head_dim,
        num_experts,
        top_k,
        norm_top_k_prob=True,
        mrope_section=(24, 20, 20),
        rope_max_wavelength=1000000,
        rope_scaling_factor=1.0,
        rope_attention_scaling=1.0,
        layer_norm_epsilon=1e-6,
        activation=None,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        sliding_window_size=None,
        router_aux_loss_coefficient=0.001,
        is_sparse_mlp=True,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.moe_intermediate_dim = moe_intermediate_dim
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_top_k_prob = norm_top_k_prob
        self.mrope_section = mrope_section
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_attention_scaling = rope_attention_scaling
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation = keras.activations.get(activation or "silu")
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.dropout_rate = dropout
        self.sliding_window_size = sliding_window_size
        self.router_aux_loss_coefficient = router_aux_loss_coefficient
        self.is_sparse_mlp = is_sparse_mlp
        self.supports_masking = True

    def build(self, input_shape):
        hidden_dim = input_shape[-1]

        # Pre-attention layer norm
        self.pre_attention_norm = Qwen3MoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_norm",
        )
        self.pre_attention_norm.build(input_shape)

        # Multi-head attention with M-RoPE
        self.attention = Qwen3OmniAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            mrope_section=self.mrope_section,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            rope_attention_scaling=self.rope_attention_scaling,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            sliding_window_size=self.sliding_window_size,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.attention.build(input_shape)

        # Post-attention layer norm
        self.post_attention_layernorm = Qwen3MoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self.post_attention_layernorm.build(input_shape)

        # MoE or dense FFN
        if self.is_sparse_mlp:
            self.sparse_moe = Qwen3SparseMoeBlock(
                hidden_dim=hidden_dim,
                moe_intermediate_dim=self.moe_intermediate_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                norm_top_k_prob=self.norm_top_k_prob,
                router_aux_loss_coefficient=self.router_aux_loss_coefficient,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dtype=self.dtype_policy,
                name="sparse_moe",
            )
            self.sparse_moe.build(input_shape)
        else:
            # Dense FFN for non-MoE layers
            self.dense_mlp = Qwen3MoeMLP(
                intermediate_dim=self.intermediate_dim,
                hidden_dim=hidden_dim,
                activation_fn="silu",
                layer_norm_epsilon=self.layer_norm_epsilon,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dtype=self.dtype_policy,
                name="dense_mlp",
            )
            self.dense_mlp.build(input_shape)

        # Dropout
        if self.dropout_rate > 0:
            self.dropout_layer = keras.layers.Dropout(
                rate=self.dropout_rate,
                dtype=self.dtype_policy,
            )

        self.built = True

    def call(
        self,
        inputs,
        position_ids=None,
        decoder_padding_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        residual = inputs
        x = self.attention(
            self.pre_attention_norm(inputs),
            position_ids=position_ids,
            attention_mask=self._compute_self_attention_mask(
                inputs=inputs,
                decoder_padding_mask=decoder_padding_mask,
                cache=cache,
                cache_update_index=cache_update_index,
            ),
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )
        if cache is not None:
            x, cache = x
        if self.dropout_rate > 0:
            x = self.dropout_layer(x, training=training)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        if self.is_sparse_mlp:
            moe_attention_mask = (
                ops.expand_dims(ops.cast(decoder_padding_mask, "bool"), axis=-1)
                if decoder_padding_mask is not None
                else None
            )
            x, _ = self.sparse_moe(
                x, attention_mask=moe_attention_mask, training=training
            )
        else:
            x = self.dense_mlp(x)
        x = x + residual

        return (x, cache) if cache is not None else x

    def _compute_self_attention_mask(
        self,
        inputs,
        decoder_padding_mask,
        cache,
        cache_update_index,
    ):
        """Combine causal and padding masks for self-attention."""
        decoder_mask = merge_padding_and_attention_mask(
            inputs, decoder_padding_mask, None
        )
        batch_size = ops.shape(inputs)[0]
        output_length = ops.shape(inputs)[1]
        # Cached decode reads the full cached key length.
        input_length = (
            ops.shape(cache)[2] if cache is not None else output_length
        )
        causal_mask = compute_causal_mask(
            batch_size,
            input_length,
            output_length,
            cache_update_index if cache_update_index is not None else 0,
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
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "norm_top_k_prob": self.norm_top_k_prob,
                "mrope_section": self.mrope_section,
                "router_aux_loss_coefficient": self.router_aux_loss_coefficient,
                "is_sparse_mlp": self.is_sparse_mlp,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_attention_scaling": self.rope_attention_scaling,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout_rate,
                "sliding_window_size": self.sliding_window_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
