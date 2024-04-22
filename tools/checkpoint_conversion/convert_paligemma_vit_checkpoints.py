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

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.paligemma.vit import PaLIGemmaViT

os.environ["KERAS_BACKEND"] = "jax"
# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def convert_vit_weights(vit_model_keras, jax_weights):
    dummy_input = np.random.rand(1, 224, 224, 3)
    vit_model_keras(dummy_input)
    num_layers = vit_model_keras.num_layers
    num_heads = vit_model_keras.num_heads
    hidden_dim = vit_model_keras.hidden_dim
    key_dim = hidden_dim // num_heads
    vit_model_keras.get_layer("classifier").weights[0].assign(
        jax_weights["img/head/kernel"]
    )
    vit_model_keras.get_layer("classifier").weights[1].assign(
        jax_weights["img/head/bias"]
    )
    for i in range(num_layers):
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            0
        ].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights[
                        "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel"
                    ][i]
                ),
                [hidden_dim, -1, hidden_dim // num_heads],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            1
        ].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights[
                        "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias"
                    ][i]
                ),
                [-1, hidden_dim // num_heads],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            2
        ].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights[
                        "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel"
                    ][i]
                ),
                [hidden_dim, -1, hidden_dim // num_heads],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            3
        ].assign(
            ops.reshape(
                jax_weights[
                    "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias"
                ][i],
                [-1, key_dim],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            4
        ].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights[
                        "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel"
                    ][i]
                ),
                [hidden_dim, -1, hidden_dim // num_heads],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            5
        ].assign(
            ops.reshape(
                jax_weights[
                    "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias"
                ][i],
                [-1, hidden_dim // num_heads],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            6
        ].assign(
            ops.reshape(
                jax_weights[
                    "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel"
                ][i],
                [-1, hidden_dim // num_heads, hidden_dim],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[i].attn.weights[
            7
        ].assign(
            ops.reshape(
                ops.squeeze(
                    jax_weights[
                        "img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias"
                    ][i]
                ),
                [-1],
            )
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[0].assign(
            jax_weights["img/Transformer/encoderblock/LayerNorm_0/scale"][i]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_1.weights[1].assign(
            jax_weights["img/Transformer/encoderblock/LayerNorm_0/bias"][i]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[0].assign(
            jax_weights["img/Transformer/encoderblock/LayerNorm_1/scale"][i]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].layer_norm_2.weights[1].assign(
            jax_weights["img/Transformer/encoderblock/LayerNorm_1/bias"][i]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[0].assign(
            jax_weights[
                "img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel"
            ][i]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_1.weights[1].assign(
            jax_weights["img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias"][
                i
            ]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[0].assign(
            jax_weights[
                "img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel"
            ][i]
        )
        vit_model_keras.get_layer("image_encoder").resblocks[
            i
        ].mlp_dense_2.weights[1].assign(
            jax_weights["img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias"][
                i
            ]
        )
    vit_model_keras.get_layer("image_encoder").encoder_layer_norm.weights[
        0
    ].assign(jax_weights["img/Transformer/encoder_norm/scale"])
    vit_model_keras.get_layer("image_encoder").encoder_layer_norm.weights[
        1
    ].assign(jax_weights["img/Transformer/encoder_norm/bias"])
    vit_model_keras.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[0].assign(
        jax_weights["img/embedding/kernel"]
    )
    vit_model_keras.get_layer(
        "image_encoder"
    ).vision_embeddings.patch_embedding.weights[1].assign(
        jax_weights["img/embedding/bias"]
    )
    vit_model_keras.get_layer(
        "image_encoder"
    ).vision_embeddings.position_embedding.weights[0].assign(
        jax_weights["img/pos_embedding"][0]
    )
    return vit_model_keras


def main(_):
    vit_model_keras = PaLIGemmaViT()
    weights = np.load("tools/checkpoint_conversion/jax_weights.npz")
    jax_weights = {}
    # Iterate over the names of the arrays stored in the .npz file
    for key in weights.files:
        # Load each array into the dictionary with its corresponding key
        jax_weights[key] = weights[key]
    vit_model_keras = convert_vit_weights(vit_model_keras, jax_weights)
    # print(jax_weights.keys())
    cow_on_beach = keras.utils.load_img(
        "tools/checkpoint_conversion/cow_beach_1.png"
    )
    cow_input = keras.utils.img_to_array(cow_on_beach)
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
