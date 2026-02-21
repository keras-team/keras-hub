"""DeepSeek V3.1 backbone model."""

import keras
from keras import ops
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_decoder_block import (
    DeepSeekV3_1DecoderBlock,
    DeepSeekV3_1RMSNorm,
)


def _deepseek_v3_1_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.DeepSeekV3_1Backbone")
class DeepSeekV3_1Backbone(Backbone):
    """DeepSeek V3.1 backbone architecture.

    Implements the full DeepSeek-V3 architecture (arXiv:2412.19437):
    - Multi-head Latent Attention (MLA) for efficient KV cache
    - DeepSeekMoE with sigmoid routing and top-K expert selection
    - First `first_k_dense_replace` layers use dense FFN; rest use MoE
    - RMSNorm pre-normalization throughout
    - YaRN RoPE for long-context extension up to 128K tokens

    Args:
        vocabulary_size: int. Vocabulary size for token embedding.
        num_layers: int. Number of transformer decoder layers. Default 61.
        hidden_dim: int. Model hidden dimension. Default 7168.
        num_query_heads: int. Number of query attention heads. Default 128.
        num_key_value_heads: int. Number of KV heads (== query heads for MLA).
        intermediate_dim: int. FFN intermediate dimension. Default 18432.
        q_lora_rank: int. Query low-rank compression dim. Default 1536.
        kv_lora_rank: int. KV low-rank compression dim (KV cache size). Default 512.
        qk_nope_head_dim: int. Per-head dim for content (non-RoPE) queries/keys.
        qk_rope_head_dim: int. Per-head dim for RoPE queries/keys. Default 64.
        v_head_dim: int. Per-head value dimension. Default 128.
        num_routed_experts: int. Total routed experts per MoE layer. Default 256.
        num_shared_experts: int. Always-active shared experts. Default 1.
        num_experts_per_tok: int. Top-K experts activated per token. Default 8.
        first_k_dense_replace: int. Layers using dense FFN before MoE. Default 3.
        rope_max_wavelength: int. RoPE base wavelength. Default 10000.
        rope_scaling_factor: float. YaRN context extension scale. Default 1.0.
        layer_norm_epsilon: float. RMSNorm epsilon. Default 1e-6.
        dropout: float. Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers=61,
        hidden_dim=7168,
        num_query_heads=128,
        num_key_value_heads=128,
        intermediate_dim=18432,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_routed_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        first_k_dense_replace=3,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            embeddings_initializer=_deepseek_v3_1_kernel_initializer(stddev=0.01),
            dtype=dtype,
            name="token_embedding",
        )

        self.transformer_layers = []
        for i in range(num_layers):
            use_moe = i >= first_k_dense_replace

            layer = DeepSeekV3_1DecoderBlock(
                hidden_dim=hidden_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_dim=intermediate_dim,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                use_moe=use_moe,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                dropout=dropout,
                kernel_initializer=_deepseek_v3_1_kernel_initializer(stddev=0.02),
                name=f"transformer_layer_{i}",
            )
            self.transformer_layers.append(layer)

        self.layer_norm = DeepSeekV3_1RMSNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="final_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )

        x = self.token_embedding(token_id_input)

        # Causal masking is handled inside each Attention layer, so we only
        # pass the padding mask here. The attention layer ANDs it with the
        # causal mask internally.
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=padding_mask_input)

        sequence_output = self.layer_norm(x)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_dim = intermediate_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

    def _build_cache(self, batch_size, sequence_length):
        """Build empty MLA KV cache for all layers.

        Each layer cache is a tuple of:
          - c_kv: (batch, seq_len, kv_lora_rank)  — compressed KV latents
          - k_rope: (batch, seq_len, qk_rope_head_dim) — decoupled RoPE keys
        """
        cache = []
        for _ in range(self.num_layers):
            c_kv = ops.zeros(
                [batch_size, sequence_length, self.kv_lora_rank],
                dtype=self.compute_dtype,
            )
            k_rope = ops.zeros(
                [batch_size, sequence_length, self.qk_rope_head_dim],
                dtype=self.compute_dtype,
            )
            cache.append((c_kv, k_rope))
        return cache

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "intermediate_dim": self.intermediate_dim,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "num_routed_experts": self.num_routed_experts,
                "num_shared_experts": self.num_shared_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "first_k_dense_replace": self.first_k_dense_replace,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config
