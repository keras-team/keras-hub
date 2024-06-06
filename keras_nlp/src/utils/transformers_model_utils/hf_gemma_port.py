# Copyright 2024 The KerasNLP Authors
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

from functools import partial

import einops
from safetensors import safe_open

from keras_nlp.src.utils.preset_utils import get_file
from keras_nlp.src.utils.preset_utils import load_config


def set_keras_weights(
    safetensor_files,
    safetensor_config,
    keras_weight,
    hf_weight_key,
    rearrange_pattern=None,
    rearrange_dims=None,
):
    safetensor_file = safetensor_files[
        safetensor_config["weight_map"][hf_weight_key]
    ]
    with safe_open(safetensor_file, framework="np") as f:
        tensor = f.get_tensor(hf_weight_key)

        if rearrange_pattern:
            if rearrange_dims:
                tensor = einops.rearrange(
                    tensor, rearrange_pattern, **rearrange_dims
                )
            else:
                tensor = einops.rearrange(tensor, rearrange_pattern)

        keras_weight.set_weights([tensor])


def load_gemma_backbone(cls, preset, load_weights):
    # get and load the backbone config
    backbone_config = load_config(preset, "config.json")

    # build a randomly initialized backbone
    backbone = cls(
        vocabulary_size=backbone_config["vocab_size"],
        num_layers=backbone_config["num_hidden_layers"],
        num_query_heads=backbone_config["num_attention_heads"],
        num_key_value_heads=backbone_config["num_key_value_heads"],
        hidden_dim=backbone_config["hidden_size"],
        intermediate_dim=backbone_config["intermediate_size"] * 2,
        head_dim=backbone_config["head_dim"],
    )

    if load_weights:
        # get and load the safetensor config
        safetensor_config = load_config(preset, "model.safetensors.index.json")

        # mapping the safetensor files to the weights
        safetensor_files = {
            fname: get_file(preset, fname)
            for fname in set(safetensor_config["weight_map"].values())
        }

        port_weight = partial(
            set_keras_weights,
            safetensor_files=safetensor_files,
            safetensor_config=safetensor_config,
        )

        # embedding
        port_weight(
            keras_weight=backbone.get_layer("token_embedding"),
            hf_weight_key="model.embed_tokens.weight",
        )

        # attention blocks
        for i in range(backbone.num_layers):
            # Norm
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).pre_attention_norm,
                hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).pre_ffw_norm,
                hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
            )

            # Attention
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).attention.query_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
                rearrange_pattern="(a c) b -> a b c",
                rearrange_dims={"a": backbone.num_query_heads},
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).attention.key_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
                rearrange_pattern="(a c) b -> a b c",
                rearrange_dims={"a": backbone.num_key_value_heads},
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).attention.value_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
                rearrange_pattern="(a c) b -> a b c",
                rearrange_dims={"a": backbone.num_key_value_heads},
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).attention.output_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
                rearrange_pattern="c (a b) -> a b c",
                rearrange_dims={"a": backbone.num_query_heads},
            )

            # MLP
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).gating_ffw,
                hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
                rearrange_pattern="b a -> a b",
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).gating_ffw_2,
                hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
                rearrange_pattern="b a -> a b",
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"decoder_block_{i}"
                ).ffw_linear,
                hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
                rearrange_pattern="b a -> a b",
            )

        # Normalization
        port_weight(
            keras_weight=backbone.get_layer("final_normalization"),
            hf_weight_key="model.norm.weight",
        )

    return backbone


def load_gemma_tokenizer(cls, preset):
    return cls(get_file(preset, "tokenizer.model"))
