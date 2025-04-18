import logging
import time
from dataclasses import dataclass
from typing import Literal

import keras
from keras import ops

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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = Embedding(args.vocab_size, args.dim)
        self.blocks = []
        for layer_id in range(args.n_layers):
            logging.info(f"Layer {layer_id}")
            self.blocks.append(
                Block(
                    layer_id,
                    args.dim,
                    args.n_heads,
                    args.q_lora_rank,
                    args.kv_lora_rank,
                    args.qk_nope_head_dim,
                    args.qk_rope_head_dim,
                    args.v_head_dim,
                    args.inter_dim,
                    args.n_dense_layers,
                    args.n_routed_experts,
                    args.n_activated_experts,
                    args.n_expert_groups,
                    args.n_limited_groups,
                    args.score_func,
                    args.route_scale,
                    args.moe_inter_dim,
                    args.n_shared_experts,
                    args.max_seq_len,
                    args.original_seq_len,
                    args.mscale,
                    args.rope_factor,
                    args.max_batch_size,
                )
            )
        self.norm = RMSNormalization(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size)
        self.freqs_cis = precompute_freqs_cis(
            args.qk_rope_head_dim,
            args.max_seq_len,
            args.beta_fast,
            args.beta_slow,
            args.rope_theta,
            args.rope_factor,
            args.original_seq_len,
        )

    def call(self, tokens, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape
                (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence
                for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = ops.shape(tokens)[1]
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = ops.full((seqlen, seqlen), float("-inf"))
            mask = ops.triu(mask, k=1)
        for layer in self.blocks:
            h = layer(h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        return logits


if __name__ == "__main__":
    keras.config.set_dtype_policy("mixed_float16")
    args = ModelArgsFull()
    x = keras.random.randint((1, 128), 0, args.vocab_size)
    logging.info("Creating model...")
    model = DeepSeekV3Backbone(args)
    logging.info(f"{model.summary()}")
    logging.info("Running dummy input...")
    outs = model(x)
    logging.info(f"{model.summary()}")
    logging.info(
        f"Output size for dummy input (shape of (1, 128)): {outs.size()}"
    )

    total_tokens_generated = 0
    total_generation_time = 0.0

    steps = 10
    logging.info(f"Generating {steps} tokens sequentially")
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
    logging.info(f"Total tokens generated: {total_tokens_generated}")
    logging.info(f"Total generation time: {total_generation_time:.2f} seconds")
    logging.info(f"Tokens per second: {tokens_per_second:.2f}")
