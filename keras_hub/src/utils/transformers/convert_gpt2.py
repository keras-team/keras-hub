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
import numpy as np

from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = GPT2Backbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["n_layer"],
        "num_heads": transformers_config["n_head"],
        "hidden_dim": transformers_config["n_embd"],
        "intermediate_dim": transformers_config["n_embd"] * 4,
        "dropout": transformers_config["resid_pdrop"],
        "max_sequence_length": transformers_config["n_positions"],
    }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="wte.weight",
    )
    loader.port_weight(
        keras_variable=backbone.position_embedding.position_embeddings,
        hf_weight_key="wpe.weight",
    )

    # Attention blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"h.{index}.ln_1.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.beta,
            hf_weight_key=f"h.{index}.ln_1.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"h.{index}.ln_2.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.beta,
            hf_weight_key=f"h.{index}.ln_2.bias",
        )

        # Attention layers
        n_embd = transformers_config["n_embd"]

        # Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.query_dense.kernel,
            hf_weight_key=f"h.{index}.attn.c_attn.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor[:, :n_embd], keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.query_dense.bias,
            hf_weight_key=f"h.{index}.attn.c_attn.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor[:n_embd], keras_shape
            ),
        )

        # Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.key_dense.kernel,
            hf_weight_key=f"h.{index}.attn.c_attn.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor[:, n_embd : 2 * n_embd], keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.key_dense.bias,
            hf_weight_key=f"h.{index}.attn.c_attn.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor[n_embd : 2 * n_embd], keras_shape
            ),
        )

        # Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.value_dense.kernel,
            hf_weight_key=f"h.{index}.attn.c_attn.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor[:, 2 * n_embd :], keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.value_dense.bias,
            hf_weight_key=f"h.{index}.attn.c_attn.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor[2 * n_embd :], keras_shape
            ),
        )

        # Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.output_dense.kernel,
            hf_weight_key=f"h.{index}.attn.c_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.output_dense.bias,
            hf_weight_key=f"h.{index}.attn.c_proj.bias",
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"h.{index}.mlp.c_fc.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"h.{index}.mlp.c_fc.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"h.{index}.mlp.c_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.bias,
            hf_weight_key=f"h.{index}.mlp.c_proj.bias",
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.gamma,
        hf_weight_key="ln_f.weight",
    )
    loader.port_weight(
        keras_variable=backbone.layer_norm.beta,
        hf_weight_key="ln_f.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    vocab_file = get_file(preset, "vocab.json")
    merges_file = get_file(preset, "merges.txt")
    return cls(
        vocabulary=vocab_file,
        merges=merges_file,
        **kwargs,
    )
