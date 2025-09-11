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
    return {
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

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

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
            hook_fn=transpose_and_reshape,
        )
        # Key
        loader.port_weight(
            keras_variable=attention_layer.key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        # Value
        loader.port_weight(
            keras_variable=attention_layer.value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        # Output
        loader.port_weight(
            keras_variable=attention_layer.output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=transpose_and_reshape,
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

        # Experts - individual expert handling
        for expert_idx in range(backbone.num_experts):
            expert = moe_block.experts
            # Gate projection
            loader.port_weight(
                keras_variable=expert.gate_up_proj[
                    expert_idx, :, : backbone.intermediate_dim
                ],
                hf_weight_key=f"model.layers.{i}.mlp.experts.{expert_idx}.gate_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=expert.gate_up_proj_bias[
                    expert_idx, : backbone.intermediate_dim
                ],
                hf_weight_key=f"model.layers.{i}.mlp.experts.{expert_idx}.gate_proj.bias",
            )
            # Up projection
            loader.port_weight(
                keras_variable=expert.gate_up_proj[
                    expert_idx, :, backbone.intermediate_dim :
                ],
                hf_weight_key=f"model.layers.{i}.mlp.experts.{expert_idx}.up_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=expert.gate_up_proj_bias[
                    expert_idx, backbone.intermediate_dim :
                ],
                hf_weight_key=f"model.layers.{i}.mlp.experts.{expert_idx}.up_proj.bias",
            )
            # Down projection
            loader.port_weight(
                keras_variable=expert.down_proj[expert_idx],
                hf_weight_key=f"model.layers.{i}.mlp.experts.{expert_idx}.down_proj.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=expert.down_proj_bias[expert_idx],
                hf_weight_key=f"model.layers.{i}.mlp.experts.{expert_idx}.down_proj.bias",
            )

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
    # For GPT-OSS, we need to extract vocabulary and
    # merges from the tokenizer.json
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

    # Convert vocabulary to the format
    # expected by BytePairTokenizer
    vocab_dict = {}
    for token, token_id in vocabulary.items():
        vocab_dict[token] = int(token_id)

    # Add special tokens from added_tokens
    for token_info in added_tokens:
        token = token_info.get("content", "")
        token_id = token_info.get("id", 0)
        vocab_dict[token] = int(token_id)

    # Convert merges from list format to
    # string format expected by BytePairTokenizer
    merges_strings = []
    for merge in merges:
        if isinstance(merge, list) and len(merge) == 2:
            merges_strings.append(f"{merge[0]} {merge[1]}")
        else:
            merges_strings.append(str(merge))

    return cls(vocabulary=vocab_dict, merges=merges_strings, **kwargs)
