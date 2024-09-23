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

from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = GemmaBackbone


def convert_backbone_config(transformers_config):
    backbone_config = dict()
    if transformers_config["model_type"] == "gemma":
        # Build Gemma backbone configuration
        backbone_config = {
            "vocabulary_size": transformers_config["vocab_size"],
            "num_layers": transformers_config["num_hidden_layers"],
            "num_query_heads": transformers_config["num_attention_heads"],
            "num_key_value_heads": transformers_config["num_key_value_heads"],
            "hidden_dim": transformers_config["hidden_size"],
            "intermediate_dim": transformers_config["intermediate_size"] * 2,
            "head_dim": transformers_config["head_dim"],
        }
    elif transformers_config["model_type"] == "gemma2":
        # Build Gemma 2 backbone configuration
        backbone_config = {
            "vocabulary_size": transformers_config["vocab_size"],
            "num_layers": transformers_config["num_hidden_layers"],
            "num_query_heads": transformers_config["num_attention_heads"],
            "num_key_value_heads": transformers_config["num_key_value_heads"],
            "hidden_dim": transformers_config["hidden_size"],
            "intermediate_dim": transformers_config["intermediate_size"] * 2,
            "head_dim": transformers_config["head_dim"],
            "query_head_dim_normalize": (
                transformers_config["head_dim"]
                == transformers_config["query_pre_attn_scalar"]
            ),
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "final_logit_soft_cap": transformers_config[
                "final_logit_softcapping"
            ],
            "attention_logit_soft_cap": transformers_config[
                "attn_logit_softcapping"
            ],
            "sliding_window_size": transformers_config["sliding_window"],
            "use_sliding_window_attention": True,
        }
    return backbone_config


def convert_weights(backbone, loader, transformers_config):
    # Embedding layer
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="model.embed_tokens.weight",
    )

    # Attention blocks
    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer.pre_attention_norm.scale,
            hf_weight_key=f"model.layers.{i}.input_layernorm.weight",
        )

        if decoder_layer.use_post_attention_norm:
            loader.port_weight(
                keras_variable=decoder_layer.post_attention_norm.scale,
                hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
            )

        if transformers_config["model_type"] == "gemma":
            loader.port_weight(
                keras_variable=decoder_layer.pre_ffw_norm.scale,
                hf_weight_key=f"model.layers.{i}.post_attention_layernorm.weight",
            )
        elif transformers_config["model_type"] == "gemma2":
            loader.port_weight(
                keras_variable=decoder_layer.pre_ffw_norm.scale,
                hf_weight_key=f"model.layers.{i}.pre_feedforward_layernorm.weight",
            )

        if decoder_layer.use_post_ffw_norm:
            loader.port_weight(
                keras_variable=decoder_layer.post_ffw_norm.scale,
                hf_weight_key=f"model.layers.{i}.post_feedforward_layernorm.weight",
            )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer.attention.query_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.key_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.value_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[0], keras_shape[2], keras_shape[1]),
                ),
                axes=(0, 2, 1),
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer.attention.output_dense.kernel,
            hf_weight_key=f"model.layers.{i}.self_attn.o_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.transpose(
                np.reshape(
                    hf_tensor,
                    (keras_shape[2], keras_shape[0], keras_shape[1]),
                ),
                axes=(1, 2, 0),
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer.gating_ffw.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.gating_ffw_2.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer.ffw_linear.kernel,
            hf_weight_key=f"model.layers.{i}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )

    # Final normalization layer
    loader.port_weight(
        keras_variable=backbone.get_layer("final_normalization").scale,
        hf_weight_key="model.norm.weight",
    )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
