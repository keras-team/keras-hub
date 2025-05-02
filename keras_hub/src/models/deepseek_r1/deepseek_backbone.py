import logging
import time
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

    print("Creating model...")
    model = DeepSeekV3Backbone.from_preset(
        "keras_hub/src/models/deepseek_r1/deepseek", load_weights=False
    )

    x = keras.random.randint((1, 128), 0, model.vocab_size)
    outs = model(x)
    print(f"{model.summary()}")
    print(f"Output size for dummy input (shape of (1, 128)): {outs.size()}")

    total_tokens_generated = 0
    total_generation_time = 0.0

    steps = 10
    print(f"Generating {steps} tokens sequentially")
    x = keras.random.randint((1, 128), 0, model.vocab_size, seed=42)

    outputs = []
    for i in tqdm(range(steps)):
        start_time = time.time()
        outs = model(x)
        res_token = outs.argmax(1).cpu().detach().numpy()[0]
        outputs.append(res_token)
        end_time = time.time() - start_time
        total_generation_time += end_time
        total_tokens_generated += 1

    tokens_per_second = total_tokens_generated / total_generation_time
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Total generation time: {total_generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Tokens: {outputs}")
