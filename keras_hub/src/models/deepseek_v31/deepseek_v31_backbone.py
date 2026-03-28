"""DeepSeek V31 backbone model."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deepseek_v31.deepseek_v31_decoder_block import (
    DeepSeekV31DecoderBlock,
)
from keras_hub.src.models.deepseek_v31.deepseek_v31_decoder_block import (
    DeepSeekV31RMSNorm,
)


def _deepseek_v31_kernel_initializer(stddev=0.006):
    return keras.initializers.RandomNormal(stddev=stddev)


@keras_hub_export("keras_hub.models.DeepSeekV31Backbone")
class DeepSeekV31Backbone(Backbone):
    """DeepSeek V31 core transformer backbone.

    Implements the full DeepSeek-V3 architecture as described in
    arXiv:2412.19437. The model uses Multi-head Latent Attention (MLA) for
    efficient KV caching, and a Mixture-of-Experts (MoE) feed-forward network
    with sigmoid-based routing in all but the first few layers.

    The first `first_k_dense_replace` layers use a dense SwiGLU feed-forward
    network; remaining layers use `DeepSeekV31MoE` with `num_routed_experts`
    total experts and `num_experts_per_tok` activated per token.

    This backbone outputs the final sequence of hidden states with shape
    `(batch_size, sequence_length, hidden_dim)`.

    Args:
        vocabulary_size: int. Size of the token vocabulary.
        num_layers: int. Number of transformer decoder layers. Defaults to
            `61`.
        hidden_dim: int. Dimensionality of hidden states. Defaults to `7168`.
        num_query_heads: int. Number of query attention heads. Defaults to
            `128`.
        num_key_value_heads: int. Number of key/value heads (equal to query
            heads for MLA). Defaults to `128`.
        intermediate_dim: int. Inner dimensionality of FFN layers. Defaults to
            `18432`.
        q_lora_rank: int. Query down-projection rank. Defaults to `1536`.
        kv_lora_rank: int. KV latent rank. Controls the per-token KV cache
            size. Defaults to `512`.
        qk_nope_head_dim: int. Per-head content (non-RoPE) dimension. Defaults
            to `128`.
        qk_rope_head_dim: int. Per-head RoPE dimension. Defaults to `64`.
        v_head_dim: int. Per-head value dimension. Defaults to `128`.
        num_routed_experts: int. Total routed MoE experts per layer. Defaults
            to `256`.
        num_shared_experts: int. Always-active shared experts per MoE layer.
            Defaults to `1`.
        num_experts_per_tok: int. Number of routed experts activated per token.
            Defaults to `8`.
        first_k_dense_replace: int. Number of initial layers that use a dense
            FFN instead of MoE. Defaults to `3`.
        rope_max_wavelength: int. RoPE base wavelength. Defaults to `10000`.
        rope_scaling_factor: float. YaRN context extension scale. Values
            greater than 1 extend the effective context length. Defaults to
            `1.0`.
        yarn_original_max_position_embeddings: int. The context length used
            during pre-training, used as the YaRN ramp reference. Defaults to
            `4096`.
        layer_norm_epsilon: float. Epsilon for RMSNorm layers. Defaults to
            `1e-6`.
        dropout: float. Dropout rate for attention and residual connections.
            Defaults to `0.0`.

    Example:

    ```python
    backbone = keras_hub.models.DeepSeekV31Backbone(
        vocabulary_size=32000,
        num_layers=4,
        hidden_dim=512,
        num_query_heads=8,
        num_key_value_heads=8,
        intermediate_dim=1024,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        num_routed_experts=8,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
    )
    token_ids = keras.random.randint((2, 16), 0, 32000)
    padding_mask = keras.ones((2, 16), dtype="bool")
    output = backbone({"token_ids": token_ids, "padding_mask": padding_mask})
    # (2, 16, 512)
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
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
        yarn_original_max_position_embeddings=4096,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        **kwargs,
    ):
        dtype = kwargs.get("dtype")
        # ===== Build sub-layers =====
        token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=False,
            embeddings_initializer=_deepseek_v31_kernel_initializer(
                stddev=0.01
            ),
            name="token_embedding",
            dtype=dtype,
        )

        transformer_layers = []
        for i in range(num_layers):
            transformer_layers.append(
                DeepSeekV31DecoderBlock(
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
                    use_moe=(i >= first_k_dense_replace),
                    rope_max_wavelength=rope_max_wavelength,
                    rope_scaling_factor=rope_scaling_factor,
                    yarn_original_max_position_embeddings=yarn_original_max_position_embeddings,  # noqa: E501
                    layer_norm_epsilon=layer_norm_epsilon,
                    dropout=dropout,
                    kernel_initializer=_deepseek_v31_kernel_initializer(
                        stddev=0.02
                    ),
                    name=f"transformer_layer_{i}",
                    dtype=dtype,
                )
            )

        layer_norm = DeepSeekV31RMSNorm(
            epsilon=layer_norm_epsilon,
            name="layer_norm",
            dtype=dtype,
        )

        # ===== Functional model =====
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )

        x = token_embedding(token_id_input)
        for layer in transformer_layers:
            x = layer(x, attention_mask=padding_mask_input)
        sequence_output = layer_norm(x)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=sequence_output,
            **kwargs,
        )

        # ===== Store attributes (must be after super().__init__) =====
        self.token_embedding = token_embedding
        self.transformer_layers = transformer_layers
        self.layer_norm = layer_norm

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
        self.yarn_original_max_position_embeddings = (
            yarn_original_max_position_embeddings
        )
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

    def _build_cache(self, token_ids):
        """Build an empty MLA KV cache for all transformer layers.

        Each layer's cache is a tuple `(c_kv, k_rope)` where:
        - `c_kv` has shape `(batch, max_len, kv_lora_rank)`
        - `k_rope` has shape `(batch, max_len, qk_rope_head_dim)`

        This is more memory-efficient than standard MHA caching, which would
        store full K and V tensors of shape `(batch, heads, max_len, head_dim)`.
        """
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        cache = []
        for _ in range(self.num_layers):
            cache.append(
                (
                    ops.zeros(
                        [batch_size, max_length, self.kv_lora_rank],
                        dtype=self.compute_dtype,
                    ),
                    ops.zeros(
                        [batch_size, max_length, self.qk_rope_head_dim],
                        dtype=self.compute_dtype,
                    ),
                )
            )
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
                "yarn_original_max_position_embeddings": (
                    self.yarn_original_max_position_embeddings
                ),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config
