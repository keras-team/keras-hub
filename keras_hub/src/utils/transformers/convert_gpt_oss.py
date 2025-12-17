"""Gpt-Oss conversion script."""

import json

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

    if (
        "head_dim" in transformers_config
        and transformers_config["head_dim"] is not None
    ):
        config["head_dim"] = transformers_config["head_dim"]

    # Include rope_scaling for YaRN support
    if (
        "rope_scaling" in transformers_config
        and transformers_config["rope_scaling"] is not None
    ):
        config["rope_scaling_factor"] = transformers_config["rope_scaling"].get(
            "factor", 32.0
        )

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
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
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
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
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
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
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
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
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

            # Decode scales first
            scales = decode_e8m0(scales)
            num_experts, out_dim, num_blocks, block_size = blocks.shape

            # Unpack 4bit values from uint8
            blocks_uint8 = blocks.astype(np.uint8)
            high_nibble = (blocks_uint8 >> 4) & 0xF
            low_nibble = blocks_uint8 & 0xF
            # Stack along new last axis
            blocks_4bit = np.stack([low_nibble, high_nibble], axis=-1)
            # Reshape to [num_experts, out_dim, num_blocks, 32 (16*2)]
            blocks_4bit = blocks_4bit.reshape(
                num_experts, out_dim, num_blocks, block_size * 2
            )

            # Decode E2M1 4bit values
            s = (blocks_4bit >> 3) & 0x1
            e = (blocks_4bit >> 1) & 0x3
            m = blocks_4bit & 0x1

            bias = 1.0
            sign = 1.0 - 2.0 * s

            normal_mask = e != 0

            values = np.empty_like(blocks_4bit, dtype=np.float32)

            values[normal_mask] = (
                sign[normal_mask]
                * (2.0 ** (e[normal_mask].astype(np.float32) - bias))
                * (1.0 + m[normal_mask].astype(np.float32) / 2.0)
            )
            values[~normal_mask] = (
                sign[~normal_mask]
                * (2.0 ** (1.0 - bias))
                * (m[~normal_mask].astype(np.float32) / 2.0)
            )

            values = values.reshape(
                num_experts, out_dim, num_blocks * block_size * 2
            )
            # Expand scales to match values shape
            scales_expanded = np.repeat(
                scales[..., np.newaxis], block_size * 2, axis=3
            )
            scales_expanded = scales_expanded.reshape(
                num_experts, out_dim, num_blocks * block_size * 2
            )
            dequantized = values * scales_expanded

            return dequantized

        # Dequantize gate_up_proj weights: [32, 90, 16, 16]
        gate_up_dequantized = dequantize_mxfp4(gate_up_blocks, gate_up_scales)

        # gate_up_dequantized: [32, 90, 256, 16]
        gate_up_proj = np.transpose(gate_up_dequantized, (0, 2, 1))

        # Dequantize down_proj weights: [32, 2880, 16, 16]
        down_dequantized = dequantize_mxfp4(down_blocks, down_scales)

        down_proj = np.transpose(down_dequantized, (0, 2, 1))

        moe_block.experts.gate_up_proj.assign(gate_up_proj)
        moe_block.experts.down_proj.assign(down_proj)

        # Load biases - reshape to match KerasHub format
        moe_block.experts.gate_up_proj_bias.assign(gate_up_bias)
        moe_block.experts.down_proj_bias.assign(down_bias)

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

    tokenizer_file = get_file(preset, "tokenizer.json")
    with open(tokenizer_file, "r") as f:
        tokenizer_data = json.load(f)
    vocabulary = tokenizer_data.get("model", {}).get("vocab", {})
    merges = tokenizer_data.get("model", {}).get("merges", [])
    added_tokens = tokenizer_data.get("added_tokens", [])

    vocab_dict = {}
    for token, token_id in vocabulary.items():
        vocab_dict[token] = int(token_id)

    for token_info in added_tokens:
        token = token_info.get("content", "")
        token_id = token_info.get("id", 0)
        vocab_dict[token] = int(token_id)

    merges_strings = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            merges_strings.append(f"{merge[0]} {merge[1]}")
        else:
            merges_strings.append(str(merge))

    return cls(vocabulary=vocab_dict, merges=merges_strings, **kwargs)
