# Copyright 2023 The KerasNLP Authors
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


def set_keras_weights_with_two_keys(
    safetensor_files,
    safetensor_config,
    keras_weight,
    hf_weight_keys,
    rearrange_patterns,
):
    list_of_tensors = list()
    for idx, (hf_weight_key, rearrange_pattern) in enumerate(
        zip(hf_weight_keys, rearrange_patterns)
    ):
        safetensor_file = safetensor_files[
            safetensor_config["weight_map"][hf_weight_key]
        ]
        with safe_open(safetensor_file, framework="np") as f:
            tensor = f.get_tensor(hf_weight_key)

            if rearrange_pattern:
                tensor = einops.rearrange(tensor, rearrange_pattern)

            list_of_tensors.append(tensor)

    keras_weight.set_weights(list_of_tensors)


def load_llama3_backbone(cls, preset, load_weights):
    # get and load the backbone config
    backbone_config = load_config(preset, "config.json")

    # build a randomly initialized backbone
    backbone = cls(
        vocabulary_size=backbone_config["vocab_size"],
        num_layers=backbone_config["num_hidden_layers"],
        num_query_heads=backbone_config["num_attention_heads"],
        hidden_dim=backbone_config["hidden_size"],
        intermediate_dim=backbone_config["intermediate_size"],
        num_key_value_heads=backbone_config["num_key_value_heads"],
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
        port_two_weight_keys = partial(
            set_keras_weights_with_two_keys,
            safetensor_files=safetensor_files,
            safetensor_config=safetensor_config,
        )

        # embedding
        port_two_weight_keys(
            keras_weight=backbone.get_layer("token_embedding"),
            hf_weight_keys=["model.embed_tokens.weight", "lm_head.weight"],
            rearrange_patterns=[None, "b a -> a b"],
        )

        # attention blocks
        for i in range(32):
            # Norm
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layernorm,
                hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._feedforward_layernorm,
                hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
            )

            # Attention
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._query_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
                rearrange_pattern="(a c) b -> b a c",
                rearrange_dims={"a": 32},
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._key_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
                rearrange_pattern="(a c) b -> b a c",
                rearrange_dims={"a": 8},
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._value_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
                rearrange_pattern="(a c) b -> b a c",
                rearrange_dims={"a": 8},
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._self_attention_layer._output_dense,
                hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
                rearrange_pattern="c (a b) -> a b c",
                rearrange_dims={"a": 32},
            )

            # MLP
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._feedforward_gate_dense,
                hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
                rearrange_pattern="b a -> a b",
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._feedforward_intermediate_dense,
                hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
                rearrange_pattern="b a -> a b",
            )
            port_weight(
                keras_weight=backbone.get_layer(
                    f"transformer_layer_{i}"
                )._feedforward_output_dense,
                hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
                rearrange_pattern="b a -> a b",
            )

        # Normalization
        port_weight(
            keras_weight=backbone.get_layer("sequence_output_layernorm"),
            hf_weight_key=f"model.norm.weight",
        )

    return backbone


def load_llama3_tokenizer(cls, preset):
    # load the tokenizer config
    tokenizer_config = load_config(preset, "tokenizer.json")

    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    bot = tokenizer_config["added_tokens"][0]  # begin of text
    eot = tokenizer_config["added_tokens"][1]  # end of text

    vocab[bot["content"]] = bot["id"]
    vocab[eot["content"]] = eot["id"]

    return cls(vocabulary=vocab, merges=merges)
