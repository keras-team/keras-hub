# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gpt-Oss conversion script."""

import numpy as np

from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = GptOssBackbone


def convert_backbone_config(transformers_config):
    """Convert a Hugging Face Gpt-Oss config to a KerasHub config."""
    config = {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "num_experts": transformers_config["num_local_experts"],
        "top_k": transformers_config["num_experts_per_tok"],
        "rope_max_wavelength": transformers_config["rope_theta"],
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        "sliding_window": transformers_config.get("sliding_window"),
        "output_router_logits": transformers_config.get(
            "output_router_logits", False
        ),
    }

    # Include head_dim in config if present in HF config
    if "head_dim" in transformers_config and transformers_config["head_dim"] is not None:
        config["head_dim"] = transformers_config["head_dim"]

    # Include rope_scaling for YaRN support
    if "rope_scaling" in transformers_config and transformers_config["rope_scaling"] is not None:
        config["rope_scaling"] = transformers_config["rope_scaling"]

    return config


def convert_weights(backbone, loader, transformers_config):
    """Convert Gpt-Oss weights."""
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.token_embedding.reverse_embeddings,
        hf_weight_key="lm_head.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )

    for i in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[i]

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer.input_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Attention layers
        attention_layer = decoder_layer.self_attention_layer
        # Query
        loader.port_weight(
            keras_variable=attention_layer.query_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, shape: np.reshape(
                np.transpose(hf_tensor, axes=(1, 0)), shape
            ),
        )
        # Query bias
        loader.port_weight(
            keras_variable=attention_layer.query_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
        )

        # Key
        loader.port_weight(
            keras_variable=attention_layer.key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, shape: np.reshape(
                np.transpose(hf_tensor, axes=(1, 0)), shape
            ),
        )
        # Key bias
        loader.port_weight(
            keras_variable=attention_layer.key_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
        )

        # Value
        loader.port_weight(
            keras_variable=attention_layer.value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, shape: np.reshape(
                np.transpose(hf_tensor, axes=(1, 0)), shape
            ),
        )
        # Value bias
        loader.port_weight(
            keras_variable=attention_layer.value_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
        )

        # Output
        loader.port_weight(
            keras_variable=attention_layer.output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=lambda hf_tensor, shape: np.reshape(
                np.transpose(hf_tensor, axes=(1, 0)), shape
            ),
        )
        # Output bias
        loader.port_weight(
            keras_variable=attention_layer.output_dense.bias,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
        )

        # Sink tokens
        loader.port_weight(
            keras_variable=attention_layer.sinks,
            hf_weight_key=f"model.layers.{i}.self_attn.sinks",
        )

        # MoE layers
        moe_block = decoder_layer.sparse_moe_block
        # Router gate
        loader.port_weight(
            keras_variable=moe_block.router.router_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.router.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=moe_block.router.router_dense.bias,
            hf_weight_key=f"model.layers.{i}.mlp.router.bias",
        )

        # Experts - handle the quantized HuggingFace MoE structure
        # The HF model uses MXFP4 quantization with _blocks and _scales

        # Get quantized weights and scales
        gate_up_blocks = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.gate_up_proj_blocks"
        )
        gate_up_scales = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.gate_up_proj_scales"
        )
        gate_up_bias = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.gate_up_proj_bias"
        )

        down_blocks = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.down_proj_blocks"
        )
        down_scales = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.down_proj_scales"
        )
        down_bias = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.down_proj_bias"
        )

        # Proper MXFP4 dequantization implementation
        def decode_e8m0(scales_8bit: np.ndarray) -> np.ndarray:
            """Decode 8-bit E8M0 floats (power-of-two scale factors)."""
            bias = 127.0
            values = 2.0 ** (scales_8bit.astype(np.float32) - bias)
            return values

        def dequantize_mxfp4(blocks, scales):
            """Dequantize MXFP4 weights (E2M1 4bit, packed in uint8)."""
            scales = decode_e8m0(scales)
            # blocks: [num_experts, out_dim, num_blocks, 16] (uint8, each value packs two 4bit numbers)
            # scales: [num_experts, out_dim, num_blocks]
            num_experts, out_dim, num_blocks, block_size = blocks.shape

            # Unpack 4bit values: each uint8 contains two 4bit values (high nibble, low nibble)
            # We'll expand last dim from 16 to 32 (each 16 uint8 -> 32 4bit values)
            # Result: [num_experts, out_dim, num_blocks, 32]
            blocks_uint8 = blocks.astype(np.uint8)
            high_nibble = (blocks_uint8 >> 4) & 0xF
            low_nibble = blocks_uint8 & 0xF
            # Stack along new last axis
            blocks_4bit = np.stack([low_nibble, high_nibble], axis=-1)
            # Reshape last two dims: [num_experts, out_dim, num_blocks, 16, 2] -> [num_experts, out_dim, num_blocks, 32]
            blocks_4bit = blocks_4bit.reshape(num_experts, out_dim, num_blocks, block_size * 2)

            # Now, decode E2M1 4bit: 1 sign bit, 2 exponent bits, 1 mantissa bit
            # Format: s e e m (bit3 bit2 bit1 bit0)
            s = (blocks_4bit >> 3) & 0x1
            e = (blocks_4bit >> 1) & 0x3
            m = blocks_4bit & 0x1

            bias = 1.0
            sign = 1.0 - 2.0*s                # +1 for s=0, -1 for s=1

            # normal numbers (e != 0)
            normal_mask = e != 0

            values = np.empty_like(blocks_4bit, dtype=np.float32)

            # normal: sign * 2^(e - bias) * (1 + m/2)
            values[normal_mask] = (
                sign[normal_mask]
                * (2.0 ** (e[normal_mask].astype(np.float32) - bias))
                * (1.0 + m[normal_mask].astype(np.float32)/2.0)
            )

            # subnormal or zero: sign * 2^(1 - bias) * (m/2)
            values[~normal_mask] = (
                sign[~normal_mask]
                * (2.0 ** (1.0 - bias))
                * (m[~normal_mask].astype(np.float32)/2.0)
            )

            # Reshape to [num_experts, out_dim, num_blocks * 32]
            values = values.reshape(num_experts, out_dim, num_blocks * block_size * 2)
            # Expand scales to match: [num_experts, out_dim, num_blocks, 1] -> [num_experts, out_dim, num_blocks, 32]
            scales_expanded = np.repeat(scales[..., np.newaxis], block_size * 2, axis=3)
            # Reshape to [num_experts, out_dim, num_blocks * 32]
            scales_expanded = scales_expanded.reshape(num_experts, out_dim, num_blocks * block_size * 2)
            # Dequantize: multiply each element by its corresponding scale
            dequantized = values * scales_expanded

            return dequantized

        # Dequantize gate_up_proj weights: [32, 5760, 90, 16] -> [32, 5760, 2880] (32 elements per block)
        gate_up_dequantized = dequantize_mxfp4(
            gate_up_blocks, gate_up_scales
        )

        # The dequantized weights need proper reshaping based on actual dimensions
        # gate_up_dequantized: [32, 5760, 2880] -> [32, hidden_dim, 2*intermediate_dim]
        # We need to transpose to [32, 2880, 5760] to get [num_experts, hidden_dim, gate+up_dim]
        gate_up_proj = np.transpose(gate_up_dequantized, (0, 2, 1))  # [32, 2880, 5760]

        # Dequantize down_proj weights: [32, 2880, 90, 16] -> [32, 2880, 2880] (32 elements per block)
        down_dequantized = dequantize_mxfp4(down_blocks, down_scales)

        # down_dequantized: [32, 2880, 2880] -> [32, intermediate_dim, hidden_dim]
        # We need to transpose to [32, 2880, 2880] to get [num_experts, hidden_dim, intermediate_dim]
        down_proj = np.transpose(down_dequantized, (0, 2, 1))  # [32, 2880, 2880]

        # Assign weights directly to the expert layer
        moe_block.experts.gate_up_proj.assign(gate_up_proj)
        moe_block.experts.down_proj.assign(down_proj)

        # Load biases - reshape to match KerasHub format
        moe_block.experts.gate_up_proj_bias.assign(
            gate_up_bias
        )  # [32, 5760]
        moe_block.experts.down_proj_bias.assign(down_bias)  # [32, 2880]

        # Post-attention layernorm
        loader.port_weight(
            keras_variable=decoder_layer.post_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key="model.norm.weight",
    )
    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    """Convert a Hugging Face tokenizer to a KerasHub tokenizer."""
    # For GPT-OSS, we need to extract vocabulary and merges from the tokenizer.json
    # and create a BytePairTokenizer
    import json

    # Get the tokenizer.json file
    tokenizer_file = get_file(preset, "tokenizer.json")

    with open(tokenizer_file, "r") as f:
        tokenizer_data = json.load(f)

    # Extract vocabulary and merges from the tokenizer.json
    vocabulary = tokenizer_data.get("model", {}).get("vocab", {})
    merges = tokenizer_data.get("model", {}).get("merges", [])
    added_tokens = tokenizer_data.get("added_tokens", [])

    # Convert vocabulary to the format expected by BytePairTokenizer
    vocab_dict = {}
    for token, token_id in vocabulary.items():
        vocab_dict[token] = int(token_id)

    # Add special tokens from added_tokens
    for token_info in added_tokens:
        token = token_info.get("content", "")
        token_id = token_info.get("id", 0)
        vocab_dict[token] = int(token_id)

    # Convert merges from list format to string format expected by BytePairTokenizer
    merges_strings = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            merges_strings.append(f"{merge[0]} {merge[1]}")
        else:
            merges_strings.append(str(merge))

    return cls(vocabulary=vocab_dict, merges=merges_strings, **kwargs)