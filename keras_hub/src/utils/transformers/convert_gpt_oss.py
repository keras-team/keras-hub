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
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").reverse_embeddings,
        hf_weight_key="lm_head.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")

        # Input layernorm
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        # Attention layers
        attention_layer = decoder_layer._self_attention_layer
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
        # Sinks
        loader.port_weight(
            keras_variable=attention_layer.sinks,
            hf_weight_key=f"model.layers.{i}.self_attn.sinks",
        )

        # MoE layers
        moe_block = decoder_layer._sparse_moe_block
        # Router gate
        loader.port_weight(
            keras_variable=moe_block._sparse_feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.router.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=moe_block._sparse_feedforward_gate_dense.bias,
            hf_weight_key=f"model.layers.{i}.mlp.router.bias",
        )

        # Batched experts
        gate_up_proj = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.gate_up_proj"
        )
        gate_up_proj_bias = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.gate_up_proj_bias"
        )
        down_proj = loader.get_tensor(f"model.layers.{i}.mlp.experts.down_proj")
        down_proj_bias = loader.get_tensor(
            f"model.layers.{i}.mlp.experts.down_proj_bias"
        )

        # De-interleave gate and up projections
        gate_proj_kernel = gate_up_proj[:, :, ::2]
        up_proj_kernel = gate_up_proj[:, :, 1::2]
        gate_proj_bias = gate_up_proj_bias[:, ::2]
        up_proj_bias = gate_up_proj_bias[:, 1::2]

        # Assign batched weights to expert_bank
        expert_bank = moe_block.expert_bank
        expert_bank._expert_feedforward_gate_dense.kernel.assign(
            gate_proj_kernel
        )
        expert_bank._expert_feedforward_gate_dense.bias.assign(gate_proj_bias)
        expert_bank._expert_feedforward_intermediate_dense.kernel.assign(
            up_proj_kernel
        )
        expert_bank._expert_feedforward_intermediate_dense.bias.assign(
            up_proj_bias
        )
        expert_bank._expert_feedforward_output_dense.kernel.assign(down_proj)
        expert_bank._expert_feedforward_output_dense.bias.assign(down_proj_bias)

        # Feedforward layernorm
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("sequence_output_layernorm").scale,
        hf_weight_key="model.norm.weight",
    )

    return backbone


def convert_tokenizer(cls, preset, **kwargs):
    """Convert a Hugging Face tokenizer to a KerasHub tokenizer."""
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
