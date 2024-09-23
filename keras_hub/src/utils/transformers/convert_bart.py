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

from keras_hub.src.models.bart.bart_backbone import BartBackbone
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = BartBackbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["encoder_attention_heads"],
        "hidden_dim": transformers_config["d_model"],
        "intermediate_dim": transformers_config["encoder_ffn_dim"],
        "dropout": transformers_config["dropout"],
        "max_sequence_length": transformers_config["max_position_embeddings"],
    }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="shared.weight",
    )
    loader.port_weight(
        keras_variable=backbone.encoder_position_embedding.position_embeddings,
        hf_weight_key="encoder.embed_positions.weight",
        hook_fn=lambda hf_tensor, keras_shape: np.reshape(
            hf_tensor[2:, :], keras_shape
        ),
    )
    loader.port_weight(
        keras_variable=backbone.decoder_position_embedding.position_embeddings,
        hf_weight_key="decoder.embed_positions.weight",
        hook_fn=lambda hf_tensor, keras_shape: np.reshape(
            hf_tensor[2:, :], keras_shape
        ),
    )

    # Encoder blocks
    for index in range(backbone.num_layers):
        encoder_layer = backbone.encoder_transformer_layers[index]
        encoder_self_attention = encoder_layer._self_attention_layer
        hf_encoder_prefix = f"encoder.layers.{index}"

        # Norm layers
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=encoder_layer._self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn_layer_norm.bias",
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_encoder_prefix}.final_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_encoder_prefix}.final_layer_norm.bias",
        )

        # Self Attention layers
        # Query
        loader.port_weight(
            keras_variable=encoder_self_attention.query_dense.kernel,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=encoder_self_attention.query_dense.bias,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.q_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Key
        loader.port_weight(
            keras_variable=encoder_self_attention.key_dense.kernel,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=encoder_self_attention.key_dense.bias,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.k_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Value
        loader.port_weight(
            keras_variable=encoder_self_attention.value_dense.kernel,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=encoder_self_attention.value_dense.bias,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.v_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Output
        loader.port_weight(
            keras_variable=encoder_self_attention.output_dense.kernel,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.out_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=encoder_self_attention.output_dense.bias,
            hf_weight_key=f"{hf_encoder_prefix}.self_attn.out_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_encoder_prefix}.fc1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_encoder_prefix}.fc1.bias",
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_encoder_prefix}.fc2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=encoder_layer._feedforward_output_dense.bias,
            hf_weight_key=f"{hf_encoder_prefix}.fc2.bias",
        )

    # Decoder blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.decoder_transformer_layers[index]
        decoder_self_attention = decoder_layer._self_attention_layer
        decoder_cross_attention = decoder_layer._cross_attention_layer
        hf_decoder_prefix = f"decoder.layers.{index}"

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn_layer_norm.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_decoder_prefix}.final_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_decoder_prefix}.final_layer_norm.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._cross_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn_layer_norm.weight",
        )
        loader.port_weight(
            keras_variable=decoder_layer._cross_attention_layer_norm.beta,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn_layer_norm.bias",
        )

        # Self Attention layers
        # Query
        loader.port_weight(
            keras_variable=decoder_self_attention.query_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_self_attention.query_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.q_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Key
        loader.port_weight(
            keras_variable=decoder_self_attention.key_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_self_attention.key_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.k_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Value
        loader.port_weight(
            keras_variable=decoder_self_attention.value_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_self_attention.value_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.v_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Output
        loader.port_weight(
            keras_variable=decoder_self_attention.output_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.out_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_self_attention.output_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.self_attn.out_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.fc1.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.fc1.bias",
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.fc2.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(hf_tensor, axes=(1, 0)),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.fc2.bias",
        )

        # Cross Attention Layers
        # Query
        loader.port_weight(
            keras_variable=decoder_cross_attention.query_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_cross_attention.query_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.q_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Key
        loader.port_weight(
            keras_variable=decoder_cross_attention.key_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_cross_attention.key_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.k_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Value
        loader.port_weight(
            keras_variable=decoder_cross_attention.value_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_cross_attention.value_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.v_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

        # Output
        loader.port_weight(
            keras_variable=decoder_cross_attention.output_dense.kernel,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.out_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_cross_attention.output_dense.bias,
            hf_weight_key=f"{hf_decoder_prefix}.encoder_attn.out_proj.bias",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor), keras_shape
            ),
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.encoder_embeddings_layer_norm.gamma,
        hf_weight_key="encoder.layernorm_embedding.weight",
    )
    loader.port_weight(
        keras_variable=backbone.encoder_embeddings_layer_norm.beta,
        hf_weight_key="encoder.layernorm_embedding.bias",
    )
    loader.port_weight(
        keras_variable=backbone.decoder_embeddings_layer_norm.gamma,
        hf_weight_key="decoder.layernorm_embedding.weight",
    )
    loader.port_weight(
        keras_variable=backbone.decoder_embeddings_layer_norm.beta,
        hf_weight_key="decoder.layernorm_embedding.bias",
    )


def convert_tokenizer(cls, preset, **kwargs):
    vocab_file = get_file(preset, "vocab.json")
    merges_file = get_file(preset, "merges.txt")
    return cls(
        vocabulary=vocab_file,
        merges=merges_file,
        **kwargs,
    )
