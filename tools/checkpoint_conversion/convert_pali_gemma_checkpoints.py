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
import os

import numpy as np
from absl import app  # noqa: E402

from keras_nlp.src.backend import keras
from keras_nlp.src.backend import ops
from keras_nlp.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)

os.environ["KERAS_BACKEND"] = "jax"
# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def print_keys(d, parent_key=""):
    for k, v in d.items():
        if isinstance(v, dict):
            if parent_key:
                print_keys(v, f"{parent_key}.{k}")
            else:
                print_keys(v, k)
        else:
            if parent_key:
                print(f"{parent_key}.{k}")
            else:
                print(k)


def get_weights_as_numpy(weights, **config):
    params_dict = {}
    num_layers_vit = config["vit_num_layers"]
    for key in weights.keys():
        if key.startswith("llm"):  # skip the Vit weights
            key_split = key.split("/")
            d = params_dict
            for k in key_split[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]

        else:
            key_split = key.split("/")

            d = params_dict
            for k in key_split[:-1]:

                if "encoderblock" == k:  # Handle encoder blocks separately
                    for block_idx in range(
                        num_layers_vit
                    ):  # Loop through 27 encoder blocks
                        block_key = "encoderblock_" + str(block_idx)
                        if block_key not in d:
                            d[block_key] = {}
                        sub_d = d[block_key]
                        for sub_key in key_split[
                            key_split.index("encoderblock") + 1 : -1
                        ]:
                            if sub_key not in sub_d:
                                sub_d[sub_key] = {}
                            sub_d = sub_d[sub_key]
                        sub_d[key_split[-1]] = np.asarray(
                            weights[key][block_idx]
                        )
                    break

                else:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
        d[key_split[-1]] = np.asarray(weights[key])
    return params_dict


def convert_pali_gemma_weights(keras_model, weights, **config):
    vit_num_layers = config["vit_num_layers"]
    vit_hidden_dim = config["vit_hidden_dim"]
    keras_model.token_embedding.embeddings.assign(
        weights["llm"]["embedder"]["input_embedding"]
    )

    for i in range(keras_model.num_layers):
        keras_model.transformer_layers[i].pre_attention_norm.scale.assign(
            weights["llm"]["layers"]["pre_attention_norm"]["scale"][i]
        )
        keras_model.transformer_layers[i].attention.query_dense.kernel.assign(
            weights["llm"]["layers"]["attn"]["q_einsum"]["w"][i]
        )
        keras_model.transformer_layers[i].attention.key_dense.kernel.assign(
            weights["llm"]["layers"]["attn"]["kv_einsum"]["w"][i][0]
        )
        keras_model.transformer_layers[i].attention.value_dense.kernel.assign(
            weights["llm"]["layers"]["attn"]["kv_einsum"]["w"][i][1]
        )
        keras_model.transformer_layers[i].attention.output_dense.kernel.assign(
            weights["llm"]["layers"]["attn"]["attn_vec_einsum"]["w"][i]
        )
        keras_model.transformer_layers[i].pre_ffw_norm.scale.assign(
            weights["llm"]["layers"]["pre_ffw_norm"]["scale"][i]
        )
        keras_model.transformer_layers[i].gating_ffw.kernel.assign(
            weights["llm"]["layers"]["mlp"]["gating_einsum"][i][0]
        )
        keras_model.transformer_layers[i].gating_ffw_2.kernel.assign(
            weights["llm"]["layers"]["mlp"]["gating_einsum"][i][1]
        )
        keras_model.transformer_layers[i].ffw_linear.kernel.assign(
            weights["llm"]["layers"]["mlp"]["linear"][i]
        )

    keras_model.layer_norm.scale.assign(weights["llm"]["final_norm"]["scale"])
    keras_model.vit_encoder.get_layer("classifier").weights[0].assign(
        weights["img"]["head"]["kernel"]
    )
    keras_model.vit_encoder.get_layer("classifier").weights[1].assign(
        weights["img"]["head"]["bias"]
    )
    for i in range(vit_num_layers):
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.key_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    weights["img"]["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["key"]["kernel"]
                ),
                [vit_hidden_dim, -1],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.key_proj.weights[1].assign(
            ops.reshape(
                ops.squeeze(
                    weights["img"]["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["key"]["bias"]
                ),
                [-1],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.query_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    weights["img"]["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["query"]["kernel"]
                ),
                [vit_hidden_dim, -1],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.query_proj.weights[1].assign(
            ops.reshape(
                weights["img"]["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["query"]["bias"],
                [-1],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.value_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    weights["img"]["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["value"]["kernel"]
                ),
                [vit_hidden_dim, -1],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.value_proj.weights[1].assign(
            ops.reshape(
                weights["img"]["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["value"]["bias"],
                [-1],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.out_proj.weights[0].assign(
            ops.reshape(
                weights["img"]["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["out"]["kernel"],
                [-1, vit_hidden_dim],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].attn.out_proj.weights[1].assign(
            ops.reshape(
                ops.squeeze(
                    weights["img"]["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["out"]["bias"]
                ),
                [-1],
            )
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[0].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["LayerNorm_0"][
                "scale"
            ]
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[1].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["LayerNorm_0"][
                "bias"
            ]
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[0].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["LayerNorm_1"][
                "scale"
            ]
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[1].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["LayerNorm_1"][
                "bias"
            ]
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[0].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_0"
            ]["kernel"]
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[1].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_0"
            ]["bias"]
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[0].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_1"
            ]["kernel"]
        )
        keras_model.vit_encoder.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[1].assign(
            weights["img"]["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_1"
            ]["bias"]
        )
    keras_model.vit_encoder.get_layer(
        "image_encoder"
    ).encoder_layer_norm.weights[0].assign(
        weights["img"]["Transformer"]["encoder_norm"]["scale"]
    )
    keras_model.vit_encoder.get_layer(
        "image_encoder"
    ).encoder_layer_norm.weights[1].assign(
        weights["img"]["Transformer"]["encoder_norm"]["bias"]
    )
    keras_model.vit_encoder.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[0].assign(
        weights["img"]["embedding"]["kernel"]
    )
    keras_model.vit_encoder.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[1].assign(
        weights["img"]["embedding"]["bias"]
    )
    keras_model.vit_encoder.get_layer(
        "image_encoder"
    ).vision_embeddings.position_embedding.weights[0].assign(
        weights["img"]["pos_embedding"][0]
    )
    return keras_model


def main(_):
    # Update config of model. please update image size according
    # to the one mentioned in checkpoints
    keras.config.set_floatx("bfloat16")
    pali_gemma_backbone_config = {
        "vit_num_layers": 27,
        "vit_hidden_dim": 1152,
        "image_size": 224,
    }
    keras_model = PaliGemmaBackbone(**pali_gemma_backbone_config)
    # This could be from kaggle or provide local dir path
    weights = np.load("tools/checkpoint_conversion/jax_weights.npz")
    jax_weights = get_weights_as_numpy(weights, **pali_gemma_backbone_config)
    keras_model = convert_pali_gemma_weights(
        keras_model, jax_weights, **pali_gemma_backbone_config
    )
    # Specify preset name
    keras_model.save_to_preset("pali_gemma_base_model")


if __name__ == "__main__":
    app.run(main)
