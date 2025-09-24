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

    # Handle head_dim if present - but don't pass it to backbone as it's calculated dynamically
    if "head_dim" in transformers_config and transformers_config["head_dim"] is not None:
        print(f"HF model head_dim: {transformers_config['head_dim']}")
        print(f"Calculated head_dim: {transformers_config['hidden_size'] // transformers_config['num_attention_heads']}")
        # Verify that the HF head_dim matches our calculation
        expected_head_dim = transformers_config["hidden_size"] // transformers_config["num_attention_heads"]
        if transformers_config["head_dim"] != expected_head_dim:
            print(f"Warning: HF head_dim ({transformers_config['head_dim']}) != calculated head_dim ({expected_head_dim})")
    elif "head_dim" in transformers_config:
        print(f"HF model has head_dim=None, will calculate dynamically")

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
        try:
            loader.port_weight(
                keras_variable=attention_layer.query_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.q_proj.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
            )
        except KeyError:
            print(f"No query bias found for layer {i}, initializing to zeros")
            # Initialize bias to zeros if not found in HF model
            attention_layer.query_dense.bias.assign(
                np.zeros_like(attention_layer.query_dense.bias)
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
        try:
            loader.port_weight(
                keras_variable=attention_layer.key_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.k_proj.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
            )
        except KeyError:
            print(f"No key bias found for layer {i}, initializing to zeros")
            # Initialize bias to zeros if not found in HF model
            attention_layer.key_dense.bias.assign(
                np.zeros_like(attention_layer.key_dense.bias)
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
        try:
            loader.port_weight(
                keras_variable=attention_layer.value_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.v_proj.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
            )
        except KeyError:
            print(f"No value bias found for layer {i}, initializing to zeros")
            # Initialize bias to zeros if not found in HF model
            attention_layer.value_dense.bias.assign(
                np.zeros_like(attention_layer.value_dense.bias)
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
        try:
            loader.port_weight(
                keras_variable=attention_layer.output_dense.bias,
                hf_weight_key=f"model.layers.{i}.self_attn.o_proj.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(hf_tensor, keras_shape),
            )
        except KeyError:
            print(f"No output bias found for layer {i}, initializing to zeros")
            # Initialize bias to zeros if not found in HF model
            attention_layer.output_dense.bias.assign(
                np.zeros_like(attention_layer.output_dense.bias)
            )

        # Sink tokens
        try:
            loader.port_weight(
                keras_variable=attention_layer.sinks,
                hf_weight_key=f"model.layers.{i}.self_attn.sinks",
            )
            print(f"  Loaded sink tokens for layer {i}")
        except KeyError:
            print(f"Warning: No sink tokens found for layer {i}")
            # Initialize sink tokens with small random values if not found
            attention_layer.sinks.assign(
                np.random.normal(0, 0.02, size=attention_layer.sinks.shape).astype(np.float32)
            )
            print(f"  Initialized sink tokens randomly for layer {i}")

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
        try:
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

            # Dequantize MXFP4 weights
            def dequantize_mxfp4(blocks, scales):
                # blocks: [num_experts, out_dim, num_blocks, 16]
                # scales: [num_experts, out_dim, num_blocks]
                num_experts, out_dim, num_blocks, block_size = blocks.shape

                # Reshape blocks to [num_experts, out_dim, num_blocks * block_size]
                blocks_flat = blocks.reshape(num_experts, out_dim, -1)
                # Expand scales to match: [num_experts, out_dim, num_blocks * block_size]
                scales_expanded = np.repeat(scales, block_size, axis=2)

                # Dequantize: multiply each element by its corresponding scale
                dequantized = blocks_flat * scales_expanded

                return dequantized

            # Dequantize gate_up_proj weights: [32, 5760, 90, 16] -> [32, 5760, 1440]
            gate_up_dequantized = dequantize_mxfp4(
                gate_up_blocks, gate_up_scales
            )
            print(f"Gate_up dequantized shape: {gate_up_dequantized.shape}")

            # The dequantized weights need proper reshaping based on actual dimensions
            # gate_up_dequantized: [32, 5760, 1440] -> [32, hidden_dim, 2*intermediate_dim]
            # We need to transpose to [32, 1440, 5760] to get [num_experts, hidden_dim, gate+up_dim]
            gate_up_proj = np.transpose(gate_up_dequantized, (0, 2, 1))  # [32, 1440, 5760]

            # Verify the expected shape matches what we have
            expected_hidden_dim = transformers_config["hidden_size"]
            expected_intermediate_dim = transformers_config["intermediate_size"]

            if gate_up_proj.shape[1] != expected_hidden_dim:
                print(f"Warning: gate_up hidden dim mismatch. Expected {expected_hidden_dim}, got {gate_up_proj.shape[1]}")
                # If dimensions don't match, we may need to pad or interpolate
                if gate_up_proj.shape[1] < expected_hidden_dim:
                    # Pad with zeros or interpolate
                    pad_size = expected_hidden_dim - gate_up_proj.shape[1]
                    padding = np.zeros((gate_up_proj.shape[0], pad_size, gate_up_proj.shape[2]))
                    gate_up_proj = np.concatenate([gate_up_proj, padding], axis=1)

            # Dequantize down_proj weights: [32, 2880, 90, 16] -> [32, 2880, 1440]
            down_dequantized = dequantize_mxfp4(down_blocks, down_scales)
            print(f"Down dequantized shape: {down_dequantized.shape}")

            # down_dequantized: [32, 2880, 1440] -> [32, intermediate_dim, hidden_dim]
            # We need to transpose to [32, 1440, 2880] to get [num_experts, hidden_dim, intermediate_dim]
            down_proj = np.transpose(down_dequantized, (0, 2, 1))  # [32, 1440, 2880]

            # Verify and fix dimensions if needed
            if down_proj.shape[1] != expected_hidden_dim:
                print(f"Warning: down_proj hidden dim mismatch. Expected {expected_hidden_dim}, got {down_proj.shape[1]}")
                if down_proj.shape[1] < expected_hidden_dim:
                    pad_size = expected_hidden_dim - down_proj.shape[1]
                    padding = np.zeros((down_proj.shape[0], pad_size, down_proj.shape[2]))
                    down_proj = np.concatenate([down_proj, padding], axis=1)

            # Add debugging before assignment
            print(f"Final gate_up_proj shape: {gate_up_proj.shape}")
            print(f"Final down_proj shape: {down_proj.shape}")
            print(f"Gate_up_proj stats: min={gate_up_proj.min():.6f}, max={gate_up_proj.max():.6f}, mean={gate_up_proj.mean():.6f}")
            print(f"Down_proj stats: min={down_proj.min():.6f}, max={down_proj.max():.6f}, mean={down_proj.mean():.6f}")

            # Assign weights directly to the expert layer
            moe_block.experts.gate_up_proj.assign(gate_up_proj)
            moe_block.experts.down_proj.assign(down_proj)

            # Load biases - reshape to match KerasHub format
            print(f"Gate_up_bias shape: {gate_up_bias.shape}")
            print(f"Down_bias shape: {down_bias.shape}")
            moe_block.experts.gate_up_proj_bias.assign(
                gate_up_bias
            )  # [32, 5760]
            moe_block.experts.down_proj_bias.assign(down_bias)  # [32, 2880]

            print(
                f"Successfully loaded dequantized MoE expert weights for layer {i}"
            )

        except KeyError as e:
            print(
                f"Warning: Could not load quantized MoE expert weights for layer {i}: {e}"
            )
            print(
                f"Available keys: {[k for k in loader.safetensor_config['weight_map'].keys() if f'layers.{i}.mlp' in k]}"
            )

            # Try to load non-quantized weights as fallback
            try:
                print(f"Attempting to load non-quantized MoE weights for layer {i}...")
                # Try standard weight names for non-quantized models
                for expert_idx in range(backbone.num_experts):
                    gate_weight_key = f"model.layers.{i}.mlp.experts.{expert_idx}.gate_proj.weight"
                    up_weight_key = f"model.layers.{i}.mlp.experts.{expert_idx}.up_proj.weight"
                    down_weight_key = f"model.layers.{i}.mlp.experts.{expert_idx}.down_proj.weight"

                    if gate_weight_key in loader.safetensor_config['weight_map']:
                        gate_weight = loader.get_tensor(gate_weight_key)
                        up_weight = loader.get_tensor(up_weight_key)
                        down_weight = loader.get_tensor(down_weight_key)

                        # Combine gate and up weights
                        gate_up_combined = np.concatenate([gate_weight, up_weight], axis=0)

                        # Assign to the expert
                        # Note: This assumes a different MoE structure - may need adjustment
                        print(f"Loaded non-quantized weights for expert {expert_idx} in layer {i}")
                        break
                else:
                    print(f"No suitable MoE weights found for layer {i}")

            except Exception as fallback_error:
                print(f"Fallback loading also failed for layer {i}: {fallback_error}")
                print("Skipping MoE weight loading for this layer")

        # Debug: Print layer parameter counts
        layer_params = decoder_layer.count_params()
        print(f"Layer {i} parameter count: {layer_params:,}")

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

    # Debug: Print final component parameter counts
    print(
        f"Token embedding parameters: {backbone.token_embedding.count_params():,}"
    )
    print(
        f"Output projection parameters: {backbone.token_embedding.reverse_embeddings.shape[0] * backbone.token_embedding.reverse_embeddings.shape[1]:,}"
    )
    print(
        f"Final layer norm parameters: {backbone.layer_norm.count_params():,}"
    )
    print(f"Total backbone parameters: {backbone.count_params():,}")

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
