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

import numpy as np

from keras_nlp.src.utils.preset_utils import HF_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import SAFETENSOR_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import get_file
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup
from keras_nlp.src.utils.preset_utils import load_config
from keras_nlp.src.utils.transformers.safetensor_utils import set_keras_weight


def load_llama3_backbone(cls, preset, load_weights):
    """
    Load and initialize the Llama3 backbone model.

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
        hidden_dim=transformers_config["hidden_size"],
        intermediate_dim=transformers_config["intermediate_size"],
        num_key_value_heads=transformers_config["num_key_value_heads"],
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

    # Embedding layers
    port_weight(
        keras_variable=backbone.get_layer("token_embedding").variables[0],
        hf_weight_key="model.embed_tokens.weight",
    )
    port_weight(
        keras_variable=backbone.get_layer("token_embedding").variables[1],
        hf_weight_key="lm_head.weight",
        # rearrange_pattern="b a -> a b",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )

    # Attention blocks
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"transformer_layer_{i}")
        # Norm layers
        port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.variables[0],
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )
        port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.variables[0],
            hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
        )

        # Attention layers
        port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.variables[
                0
            ],
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            # rearrange_patterns="(b c) a -> a b c,
            # rearrange_dims={"b": backbone.num_query_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[1], keras_shape[2], keras_shape[0]),
                ),
                axes=(2, 0, 1),
            ),
        )
        port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.variables[
                0
            ],
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            # rearrange_patterns="(b c) a -> a b c",
            # rearrange_dims={"b": backbone.num_key_value_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[1], keras_shape[2], keras_shape[0]),
                ),
                axes=(2, 0, 1),
            ),
        )
        port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.variables[
                0
            ],
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            # rearrange_patterns="(b c) a -> a b c",
            # rearrange_dims={"b": backbone.num_key_value_heads},
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[1], keras_shape[2], keras_shape[0]),
                ),
                axes=(2, 0, 1),
            ),
        )
        port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.variables[
                0
            ],
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
            keras_variable=decoder_layer._feedforward_gate_dense.variables[0],
            hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.variables[
                0
            ],
            hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.variables[0],
            hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
            # rearrange_patterns="b a -> a b",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # Final normalization layer
    port_weight(
        keras_variable=backbone.get_layer(
            "sequence_output_layernorm"
        ).variables[0],
        hf_weight_key="model.norm.weight",
    )

    return backbone


def load_llama3_tokenizer(cls, preset):
    """
    Load the Llama3 tokenizer.

    Args:
        cls (class): Tokenizer class.
        preset (str): Preset configuration name.

    Returns:
        tokenizer: Initialized tokenizer.
    """
    tokenizer_config = load_config(preset, "tokenizer.json")
    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    bot = tokenizer_config["added_tokens"][0]  # begin of text
    eot = tokenizer_config["added_tokens"][1]  # end of text

    vocab[bot["content"]] = bot["id"]
    vocab[eot["content"]] = eot["id"]

    return cls(vocabulary=vocab, merges=merges)
