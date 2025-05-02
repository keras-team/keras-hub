import logging
import time
from dataclasses import dataclass
from typing import Literal

import keras
from keras import ops
from tqdm import tqdm

from keras_hub.src.api_export import keras_hub_export

# TODO: Replace with keras.layers.RMSNormalization
from keras_hub.src.layers.modeling.rms_normalization import RMSNormalization
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deepseek_r1.deepseek_layers import Block
from keras_hub.src.models.deepseek_r1.deepseek_layers import (
    ColumnParallelLinear,
)
from keras_hub.src.models.deepseek_r1.deepseek_layers import Embedding
from keras_hub.src.models.deepseek_r1.deepseek_layers import (
    precompute_freqs_cis,
)

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"




@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    # n_layers: int = 27
    n_layers: int = 1
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


@dataclass
class ModelArgsFull:
    max_batch_size: int = 1
    max_seq_len: int = 163840
    vocab_size: int = 129280
    dim: int = 7168
    inter_dim: int = 18432
    moe_inter_dim: int = 2048
    n_layers: int = 61
    n_dense_layers: int = 1
    n_heads: int = 128
    # moe
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 8  # do we need this?
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_scale: float = 2.5
    # mla
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0


@keras_hub_export("keras_hub.models.DeepSeekV3Backbone")
class DeepSeekV3Backbone(Backbone):
    """A DeepSeekV3 encoder network.

    This network implements a transformer-based encoder with configurable parameters,
    including rotary embeddings, experts, and dense layers. It includes the embedding lookups,
    transformer blocks, normalization layers, and a linear head.

    The default constructor gives a fully customizable, randomly initialized DeepSeekV3 encoder
    with any number of layers, heads, and embedding dimensions. To load preset architectures and weights,
    use the `from_preset()` constructor.

    Args:
        vocab_size: int. The size of the token vocabulary.
        n_layers: int. The number of transformer blocks.
        dim: int. The dimension of the embedding and transformer layers.
        n_heads: int. The number of attention heads for each transformer block.
        q_lora_rank: int. Rank for Q LoRA.
        kv_lora_rank: int. Rank for KV LoRA.
        inter_dim: int. The intermediate dimension for the dense layers.
        n_dense_layers: int. Number of dense layers.
        n_routed_experts: int. Number of routed experts in the MoE blocks.
        n_activated_experts: int. Number of activated experts in the MoE blocks.
        n_expert_groups: int. Number of expert groups.
        n_limited_groups: int. Number of limited groups.
        score_func: str. The scoring function for the routing mechanism.
        route_scale: float. The scale for routing probabilities.
        max_seq_len: int. The maximum sequence length the model can handle.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype for model computations and weights.

    Examples:
    ```python
    input_data = {
        "token_ids": np.ones(shape=(1, 12), dtype="int32"),
    }

    model = keras_hub.models.DeepSeekV3Backbone(
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        max_batch_size,
        max_seq_len,
        vocab_size,
        dim,
        inter_dim,
        moe_inter_dim,
        n_layers,
        n_dense_layers,
        n_heads,
        # moe
        n_routed_experts,
        n_shared_experts,
        n_activated_experts,
        n_expert_groups,
        n_limited_groups,
        score_func,
        route_scale,
        # mla
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        # yarn
        original_seq_len,
        rope_theta,
        rope_factor,
        beta_fast,
        beta_slow,
        mscale,
        **kwargs,
    ):
        # === Layers ===
        self.embedding_layer = Embedding(vocab_size, dim)

        # DeepSeekV3 transformer blocks (stack of layers)
        self.blocks = []
        for layer_id in range(n_layers):
            self.blocks.append(
                Block(
                    layer_id,
                    dim,
                    n_heads,
                    q_lora_rank,
                    kv_lora_rank,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    v_head_dim,
                    inter_dim,
                    n_dense_layers,
                    n_routed_experts,
                    n_activated_experts,
                    n_expert_groups,
                    n_limited_groups,
                    score_func,
                    route_scale,
                    moe_inter_dim,
                    n_shared_experts,
                    max_seq_len,
                    original_seq_len,
                    mscale,
                    rope_factor,
                    max_batch_size,
                )
            )

        # Layer normalization and output head
        self.norm = RMSNormalization(dim)
        self.head = ColumnParallelLinear(dim, vocab_size)

        # Precompute freqs_cis for rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            qk_rope_head_dim,
            max_seq_len,
            beta_fast,
            beta_slow,
            rope_theta,
            rope_factor,
            original_seq_len,
        )

        # === Functional Model ===
        tokens = keras.Input(shape=(128,), dtype="int32", name="tokens")
        print(ops.shape(tokens))

        seqlen = ops.shape(tokens)[1]

        h = self.embedding_layer(tokens)
        freqs_cis = self.freqs_cis[0 : 0 + seqlen]
        mask = None
        if seqlen > 1:
            mask = ops.full((seqlen, seqlen), float("-inf"))
            mask = ops.triu(mask, k=1)
        for layer in self.blocks:
            h = layer(h, start_pos=0, freqs_cis=freqs_cis, mask=mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)

        super().__init__(
            inputs={"tokens": tokens},
            outputs=logits,
            **kwargs,
        )

        # === Config ===
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.inter_dim = inter_dim
        self.n_dense_layers = n_dense_layers
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_expert_groups = n_expert_groups
        self.n_limited_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        self.max_seq_len = max_seq_len

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "n_layers": self.n_layers,
                "dim": self.dim,
                "n_heads": self.n_heads,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "inter_dim": self.inter_dim,
                "n_dense_layers": self.n_dense_layers,
                "n_routed_experts": self.n_routed_experts,
                "n_activated_experts": self.n_activated_experts,
                "n_expert_groups": self.n_expert_groups,
                "n_limited_groups": self.n_limited_groups,
                "score_func": self.score_func,
                "route_scale": self.route_scale,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    #keras.config.set_dtype_policy("mixed_float16")
    args = ModelArgs()
    x = keras.random.randint((1, 128), 0, args.vocab_size)
    print("Creating model...")
    model = DeepSeekV3Backbone(
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        dim=args.dim,
        inter_dim=args.inter_dim,
        moe_inter_dim=args.moe_inter_dim,
        n_layers=args.n_layers,
        n_dense_layers=args.n_dense_layers,
        n_heads=args.n_heads,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        n_expert_groups=args.n_expert_groups,
        n_limited_groups=args.n_limited_groups,
        score_func=args.score_func,
        route_scale=args.route_scale,
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        original_seq_len=args.original_seq_len,
        rope_theta=args.rope_theta,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
        mscale=args.mscale,
    )
    outs = model(x)
    print(f"{model.summary()}")
    print(
        f"Output size for dummy input (shape of (1, 128)): {outs.size()}"
    )

    total_tokens_generated = 0
    total_generation_time = 0.0

    steps = 10
    print(f"Generating {steps} tokens sequentially")
    x = keras.random.randint((1, 128), 0, args.vocab_size, seed=42)

    for i in tqdm(range(steps)):
        start_time = time.time()
        outs = model(x)
        res_token = outs.argmax(1).unsqueeze(0)
        x = ops.concatenate([x, res_token], 1)
        end_time = time.time() - start_time
        total_generation_time += end_time
        total_tokens_generated += 1

    tokens_per_second = total_tokens_generated / total_generation_time
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Total generation time: {total_generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
