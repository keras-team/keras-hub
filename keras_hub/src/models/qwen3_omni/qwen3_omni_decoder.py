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
from keras_hub.src.models.qwen3_moe.qwen3_moe_decoder import (
    compute_load_balancing_loss,
)
from keras_hub.src.models.qwen3_moe.qwen3_moe_layernorm import Qwen3MoeLayerNorm
from keras_hub.src.models.qwen3_omni.qwen3_omni_attention import (
    Qwen3OmniAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen3OmniTransformerDecoder(keras.layers.Layer):
    """Qwen3-Omni transformer decoder block with MoE.

    This decoder block combines:
    - Multi-head attention with M-RoPE (Qwen3OmniAttention)
    - Mixture-of-Experts feedforward network (from Qwen3MoE)
    - Pre-normalization architecture
    - Residual connections

    Args:
        intermediate_dim: int. Dimension of dense FFN
            (used when MoE is disabled).
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads (for GQA).
        moe_intermediate_dim: int. Intermediate dimension for each MoE expert.
        head_dim: int. Dimension of each attention head.
        num_experts: int. Total number of experts in the MoE layer.
        top_k: int. Number of experts to activate per token.
        norm_top_k_prob: bool. Whether to normalize top-k probabilities.
        mrope_section: tuple. M-RoPE section dimensions
            [text, temporal, spatial].
        rope_max_wavelength: int. Maximum wavelength for M-RoPE.
        rope_scaling_factor: float. Scaling factor for M-RoPE.
        rope_attention_scaling: float. Attention scaling for M-RoPE
            (default 1.0).
        layer_norm_epsilon: float. Epsilon for layer normalization.
        activation: callable. Activation function (typically SiLU).
        kernel_initializer: initializer. Kernel initializer.
        dropout: float. Dropout rate.
        sliding_window_size: int or None. Size of sliding attention window.
        router_aux_loss_coefficient: float. Auxiliary loss coefficient for MoE.
        is_sparse_mlp: bool. Whether to use sparse MoE or dense FFN.
        dtype: DType policy for the layer.
        **kwargs: Additional layer arguments.
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
        self.activation = activation or ops.silu
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.dropout_rate = dropout
        self.sliding_window_size = sliding_window_size
        self.router_aux_loss_coefficient = router_aux_loss_coefficient
        self.is_sparse_mlp = is_sparse_mlp

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

        # Pre-FFN layer norm
        self.pre_ffw_norm = Qwen3MoeLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_ffw_norm",
        )
        self.pre_ffw_norm.build(input_shape)

        # MoE or dense FFN
        if self.is_sparse_mlp:
            # Sparse MoE feedforward (reuse from Qwen3MoE)
            self.sparse_moe = Qwen3SparseMoeBlock(
                hidden_dim=hidden_dim,
                moe_intermediate_dim=self.moe_intermediate_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                norm_top_k_prob=self.norm_top_k_prob,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dtype=self.dtype_policy,
                name="sparse_moe",
            )
            self.sparse_moe.build(input_shape)
        else:
            # Dense FFN (for non-MoE layers)
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
        """Forward pass of the decoder block.

        Args:
            inputs: Input tensor of shape (batch, seq_len, hidden_dim).
            position_ids: Position IDs for M-RoPE, shape (3, batch, seq_len).
            decoder_padding_mask: Padding mask for attention.
            cache: KV cache for generation (optional).
            cache_update_index: Index for cache update.
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        x = inputs

        # Create attention mask from padding mask
        attention_mask = None
        if decoder_padding_mask is not None:
            batch_size = ops.shape(inputs)[0]
            input_length = ops.shape(inputs)[1]
            attention_mask = merge_padding_and_attention_mask(
                inputs,
                decoder_padding_mask,
                compute_causal_mask(
                    batch_size,
                    input_length,
                    input_length,
                    cache_update_index or 0,
                ),
            )

        # Self-attention block with residual
        residual = x
        x = self.pre_attention_norm(x)

        attention_output = self.attention(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )

        # Handle cache return
        if cache is not None:
            attention_output, cache = attention_output

        if self.dropout_rate > 0:
            attention_output = self.dropout_layer(
                attention_output, training=training
            )

        x = residual + attention_output

        # Feedforward block with residual
        residual = x
        x = self.pre_ffw_norm(x)

        if self.is_sparse_mlp:
            # MoE feedforward
            x, router_logits = self.sparse_moe(x)

            # Compute auxiliary loss for load balancing
            if training:
                aux_loss = compute_load_balancing_loss(
                    router_logits,
                    self.num_experts,
                    self.top_k,
                    attention_mask,
                )
                self.add_loss(self.router_aux_loss_coefficient * aux_loss)
        else:
            # Dense feedforward
            x = self.dense_mlp(x)

        if self.dropout_rate > 0:
            x = self.dropout_layer(x, training=training)

        x = residual + x

        if cache is not None:
            return x, cache
        return x

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
