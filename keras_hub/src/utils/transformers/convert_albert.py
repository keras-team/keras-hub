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

from keras_hub.src.models.albert.albert_backbone import AlbertBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = AlbertBackbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "embedding_dim": transformers_config["embedding_size"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_groups": transformers_config["num_hidden_groups"],
        "num_inner_repetitions": transformers_config["inner_group_num"],
        "dropout": transformers_config["attention_probs_dropout_prob"],
        "max_sequence_length": transformers_config["max_position_embeddings"],
        "num_segments": transformers_config["type_vocab_size"],
    }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="albert.embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.position_embedding.position_embeddings,
        hf_weight_key="albert.embeddings.position_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.segment_embedding.embeddings,
        hf_weight_key="albert.embeddings.token_type_embeddings.weight",
    )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.gamma,
        hf_weight_key="albert.embeddings.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.beta,
        hf_weight_key="albert.embeddings.LayerNorm.bias",
    )

    # Encoder Embeddings
    loader.port_weight(
        keras_variable=backbone.embeddings_projection.kernel,
        hf_weight_key="albert.encoder.embedding_hidden_mapping_in.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_projection.bias,
        hf_weight_key="albert.encoder.embedding_hidden_mapping_in.bias",
    )

    # Encoder Group Layers
    for group_idx in range(backbone.num_groups):
        for inner_layer_idx in range(backbone.num_inner_repetitions):
            keras_group = backbone.get_layer(
                f"group_{group_idx}_inner_layer_{inner_layer_idx}"
            )
            hf_group_prefix = (
                "albert.encoder.albert_layer_groups."
                f"{group_idx}.albert_layers.{inner_layer_idx}."
            )

            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.query_dense.kernel,
                hf_weight_key=f"{hf_group_prefix}attention.query.weight",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    np.transpose(hf_tensor), keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.query_dense.bias,
                hf_weight_key=f"{hf_group_prefix}attention.query.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    hf_tensor, keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.key_dense.kernel,
                hf_weight_key=f"{hf_group_prefix}attention.key.weight",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    np.transpose(hf_tensor), keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.key_dense.bias,
                hf_weight_key=f"{hf_group_prefix}attention.key.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    hf_tensor, keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.value_dense.kernel,
                hf_weight_key=f"{hf_group_prefix}attention.value.weight",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    np.transpose(hf_tensor), keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.value_dense.bias,
                hf_weight_key=f"{hf_group_prefix}attention.value.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    hf_tensor, keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.output_dense.kernel,
                hf_weight_key=f"{hf_group_prefix}attention.dense.weight",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    np.transpose(hf_tensor), keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer.output_dense.bias,
                hf_weight_key=f"{hf_group_prefix}attention.dense.bias",
                hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                    hf_tensor, keras_shape
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer_norm.gamma,
                hf_weight_key=f"{hf_group_prefix}attention.LayerNorm.weight",
            )
            loader.port_weight(
                keras_variable=keras_group._self_attention_layer_norm.beta,
                hf_weight_key=f"{hf_group_prefix}attention.LayerNorm.bias",
            )
            loader.port_weight(
                keras_variable=keras_group._feedforward_intermediate_dense.kernel,
                hf_weight_key=f"{hf_group_prefix}ffn.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._feedforward_intermediate_dense.bias,
                hf_weight_key=f"{hf_group_prefix}ffn.bias",
            )
            loader.port_weight(
                keras_variable=keras_group._feedforward_output_dense.kernel,
                hf_weight_key=f"{hf_group_prefix}ffn_output.weight",
                hook_fn=lambda hf_tensor, _: np.transpose(
                    hf_tensor, axes=(1, 0)
                ),
            )
            loader.port_weight(
                keras_variable=keras_group._feedforward_output_dense.bias,
                hf_weight_key=f"{hf_group_prefix}ffn_output.bias",
            )
            loader.port_weight(
                keras_variable=keras_group._feedforward_layer_norm.gamma,
                hf_weight_key=f"{hf_group_prefix}full_layer_layer_norm.weight",
            )
            loader.port_weight(
                keras_variable=keras_group._feedforward_layer_norm.beta,
                hf_weight_key=f"{hf_group_prefix}full_layer_layer_norm.bias",
            )

    # Pooler
    loader.port_weight(
        keras_variable=backbone.pooled_dense.kernel,
        hf_weight_key="albert.pooler.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
    )
    loader.port_weight(
        keras_variable=backbone.pooled_dense.bias,
        hf_weight_key="albert.pooler.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    return cls(get_file(preset, "spiece.model"), **kwargs)
