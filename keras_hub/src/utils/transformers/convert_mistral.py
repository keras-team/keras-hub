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

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = MistralBackbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "rope_max_wavelength": transformers_config["rope_theta"],
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        "sliding_window": transformers_config["sliding_window"],
    }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="model.embed_tokens.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
    )
    loader.port_weight(
        keras_variable=backbone.token_embedding.reverse_embeddings,
        hf_weight_key="lm_head.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(
            hf_tensor.astype(np.float16), axes=(1, 0)
        ),
    )

    # Attention blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{index}.input_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{index}.post_attention_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.o_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key="model.norm.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
    )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
