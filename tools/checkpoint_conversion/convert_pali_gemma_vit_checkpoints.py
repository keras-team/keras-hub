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
from keras_nlp.src.models.pali_gemma.pali_gemma_vit import PaliGemmaVit

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


def get_weights_as_numpy(weights):
    params_dict = {}
    num_layers = 27
    for key in weights.keys():
        if key.startswith("llm"):  # skip the Vit weights
            continue
        key_split = key.split("/")

        d = params_dict
        for k in key_split[:-1]:
            if k == "img":
                k = "params"

            if "encoderblock" == k:  # Handle encoder blocks separately
                for block_idx in range(
                    num_layers
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
                    sub_d[key_split[-1]] = np.asarray(weights[key][block_idx])
                break

            else:
                if k not in d:
                    d[k] = {}
                d = d[k]
        d[key_split[-1]] = np.asarray(weights[key])
    print_keys(params_dict)
    return params_dict


def convert_vit_weights(vit_model_keras, jax_weights):
    num_layers = vit_model_keras.num_layers
    hidden_dim = vit_model_keras.hidden_dim
    vit_model_keras.get_layer("classifier").weights[0].assign(
        jax_weights["head"]["kernel"]
    )
    vit_model_keras.get_layer("classifier").weights[1].assign(
        jax_weights["head"]["bias"]
    )
    for i in range(num_layers):
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.key_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["key"]["kernel"]
                ),
                [hidden_dim, -1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.key_proj.weights[1].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["key"]["bias"]
                ),
                [-1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.query_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["query"]["kernel"]
                ),
                [hidden_dim, -1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.query_proj.weights[1].assign(
            ops.reshape(
                jax_weights["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["query"]["bias"],
                [-1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.value_proj.weights[0].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["value"]["kernel"]
                ),
                [hidden_dim, -1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.value_proj.weights[1].assign(
            ops.reshape(
                jax_weights["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["value"]["bias"],
                [-1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.out_proj.weights[0].assign(
            ops.reshape(
                jax_weights["Transformer"][f"encoderblock_{i}"][
                    "MultiHeadDotProductAttention_0"
                ]["out"]["kernel"],
                [-1, hidden_dim],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].attn.out_proj.weights[1].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights["Transformer"][f"encoderblock_{i}"][
                        "MultiHeadDotProductAttention_0"
                    ]["out"]["bias"]
                ),
                [-1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[0].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["LayerNorm_0"][
                "scale"
            ]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[1].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["LayerNorm_0"][
                "bias"
            ]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[0].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["LayerNorm_1"][
                "scale"
            ]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[1].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["LayerNorm_1"][
                "bias"
            ]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[0].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_0"
            ]["kernel"]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[1].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_0"
            ]["bias"]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[0].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_1"
            ]["kernel"]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[1].assign(
            jax_weights["Transformer"][f"encoderblock_{i}"]["MlpBlock_0"][
                "Dense_1"
            ]["bias"]
        )
    vit_model_keras.get_layer("image_encoder").encoder_layer_norm.weights[
        0
    ].assign(jax_weights["Transformer"]["encoder_norm"]["scale"])
    vit_model_keras.get_layer("image_encoder").encoder_layer_norm.weights[
        1
    ].assign(jax_weights["Transformer"]["encoder_norm"]["bias"])
    vit_model_keras.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[0].assign(
        jax_weights["embedding"]["kernel"]
    )
    vit_model_keras.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[1].assign(
        jax_weights["embedding"]["bias"]
    )
    vit_model_keras.get_layer(
        "image_encoder"
    ).vision_embeddings.position_embedding.weights[0].assign(
        jax_weights["pos_embedding"][0]
    )
    return vit_model_keras


def main(_):
    vit_model_keras = PaliGemmaVit(image_resolution=224)
    weights = np.load("tools/checkpoint_conversion/jax_weights.npz")
    jax_weights = get_weights_as_numpy(weights)
    vit_model_keras = convert_vit_weights(
        vit_model_keras, jax_weights["params"]
    )
    # print(jax_weights.keys())
    cow_on_beach = keras.utils.load_img(
        "tools/checkpoint_conversion/cow_beach_1.png"
    )
    cow_input = keras.utils.img_to_array(cow_on_beach) / 127.5 - 1
    cow_input = ops.expand_dims(cow_input, axis=0)
    vit_output = vit_model_keras([cow_input])
    print("output of vit model : ", vit_output)
    expected_output = np.load("tools/checkpoint_conversion/intermediates.npz")
    jax_numerics = {}
    for key in expected_output.files:
        jax_numerics[key] = expected_output[key]
    print("Expected output : ", jax_numerics["img/zimg"])


if __name__ == "__main__":
    app.run(main)
