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
import numpy as np

from keras_nlp.src.utils.preset_utils import HF_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import get_file
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup
from keras_nlp.src.utils.preset_utils import load_config
from keras_nlp.src.utils.transformers.safetensor_utils import SafetensorLoader


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["n_layers"],
        "num_heads": transformers_config["n_heads"],
        "hidden_dim": transformers_config["dim"],
        "intermediate_dim": transformers_config["hidden_dim"],
        "dropout": transformers_config["dropout"],
        "max_sequence_length": transformers_config["max_position_embeddings"],
    }


def convert_weights(backbone, loader):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.get_layer(
            "token_and_position_embedding"
        ).token_embedding.embeddings,
        hf_weight_key="distilbert.embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer(
            "token_and_position_embedding"
        ).position_embedding.position_embeddings,
        hf_weight_key="distilbert.embeddings.position_embeddings.weight",
    )

    # Attention blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"distilbert.transformer.layer.{index}.sa_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.beta,
            hf_weight_key=f"distilbert.transformer.layer.{index}.sa_layer_norm.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"distilbert.transformer.layer.{index}.output_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.beta,
            hf_weight_key=f"distilbert.transformer.layer.{index}.output_layer_norm.bias",
        )

        # Attention layers
        # Query
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.query_dense.kernel,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.q_lin.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.query_dense.bias,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.q_lin.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
        )

        # Key
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.key_dense.kernel,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.k_lin.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.key_dense.bias,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.k_lin.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
        )

        # Value
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.value_dense.kernel,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.v_lin.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.value_dense.bias,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.v_lin.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                hf_tensor, keras_shape
            ),
        )

        # Output
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.output_dense.kernel,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.out_lin.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer.output_dense.bias,
            hf_weight_key=f"distilbert.transformer.layer.{index}.attention.out_lin.bias",
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"distilbert.transformer.layer.{index}.ffn.lin1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"distilbert.transformer.layer.{index}.ffn.lin1.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"distilbert.transformer.layer.{index}.ffn.lin2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.bias,
            hf_weight_key=f"distilbert.transformer.layer.{index}.ffn.lin2.bias",
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.gamma,
        hf_weight_key="distilbert.embeddings.LayerNorm.weight",
    )
    loader.port_weight(
        keras_variable=backbone.embeddings_layer_norm.beta,
        hf_weight_key="distilbert.embeddings.LayerNorm.bias",
    )

    return backbone


def load_distilbert_backbone(cls, preset, load_weights):
    transformers_config = load_config(preset, HF_CONFIG_FILE)
    keras_config = convert_backbone_config(transformers_config)
    backbone = cls(**keras_config)
    if load_weights:
        jax_memory_cleanup(backbone)
        with SafetensorLoader(preset) as loader:
            convert_weights(backbone, loader)
    return backbone


def load_distilbert_tokenizer(cls, preset):
    transformers_config = load_config(preset, HF_TOKENIZER_CONFIG_FILE)
    return cls(
        get_file(preset, "vocab.txt"),
        lowercase=transformers_config["do_lower_case"],
    )
