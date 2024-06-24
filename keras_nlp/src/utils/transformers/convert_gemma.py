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

import numpy as np

from keras_nlp.src.utils.preset_utils import HF_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import SAFETENSOR_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import get_file
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup
from keras_nlp.src.utils.preset_utils import load_config
from keras_nlp.src.utils.transformers.safetensor_utils import set_keras_weight


def load_gemma_backbone(cls, preset, load_weights):
    """
    Load and initialize the Gemma backbone model.

    Args:
        cls (class): Keras model class.
        preset (str): Preset configuration name.
        load_weights (bool): Whether to load the weights.

    Returns:
        backbone: Initialized Keras model backbone.
    """
    transformers_config = load_config(preset, HF_CONFIG_FILE)

    backbone = cls(
        vocabulary_size=transformers_config["vocab_size"],
        num_layers=transformers_config["num_hidden_layers"],
        num_query_heads=transformers_config["num_attention_heads"],
        num_key_value_heads=transformers_config["num_key_value_heads"],
        hidden_dim=transformers_config["hidden_size"],
        intermediate_dim=transformers_config["intermediate_size"] * 2,
        head_dim=transformers_config["head_dim"],
    )

    if not load_weights:
        return backbone

    jax_memory_cleanup(backbone)
    # Code to port the weights from safetensors into the keras nlp model
    safetensor_config = load_config(preset, SAFETENSOR_CONFIG_FILE)
    safetensor_files = {
        fname: get_file(preset, fname)
        for fname in set(safetensor_config["weight_map"].values())
    }
    port_weight = partial(
        set_keras_weight,
        safetensor_files=safetensor_files,
        safetensor_config=safetensor_config,
    )

    # Embedding layer
    port_weight(
        keras_variable=backbone.get_layer("token_embedding").variables[0],
        hf_weight_key="model.embed_tokens.weight",
    )

    # Attention blocks
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        # Norm layers
        port_weight(
            keras_variable=decoder_layer.pre_attention_norm.variables[0],
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )
        port_weight(
            keras_variable=decoder_layer.pre_ffw_norm.variables[0],
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

        # Attention layers
        port_weight(
            keras_variable=decoder_layer.attention.query_dense.variables[0],
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            # rearrange_patterns="(a c) b -> a b c",
            # rearrange_dims={"a": backbone.num_query_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        port_weight(
            keras_variable=decoder_layer.attention.key_dense.variables[0],
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            # rearrange_patterns="(a c) b -> a b c",
            # rearrange_dims={"a": backbone.num_key_value_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        port_weight(
            keras_variable=decoder_layer.attention.value_dense.variables[0],
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            # rearrange_patterns="(a c) b -> a b c",
            # rearrange_dims={"a": backbone.num_key_value_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        port_weight(
            keras_variable=decoder_layer.attention.output_dense.variables[0],
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            # rearrange_patterns="c (a b) -> a b c",
            # rearrange_dims={"a": backbone.num_query_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[2], keras_shape[0], keras_shape[1]),
                ),
                axes=(1, 2, 0),
            ),
        )

        # MLP layers
        port_weight(
            keras_variable=decoder_layer.gating_ffw.variables[0],
            hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        port_weight(
            keras_variable=decoder_layer.gating_ffw_2.variables[0],
            hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        port_weight(
            keras_variable=decoder_layer.ffw_linear.variables[0],
            hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # Final normalization layer
    port_weight(
        keras_variable=backbone.get_layer("final_normalization").variables[0],
        hf_weight_key="model.norm.weight",
    )

    return backbone


def load_gemma_tokenizer(cls, preset):
    """
    Load the Gemma tokenizer.

    Args:
        cls (class): Tokenizer class.
        preset (str): Preset configuration name.

    Returns:
        tokenizer: Initialized tokenizer.
    """
    return cls(get_file(preset, "tokenizer.model"))
