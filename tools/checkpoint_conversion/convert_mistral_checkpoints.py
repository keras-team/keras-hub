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
import json
import pathlib

import torch

from keras_nlp.models import MistralBackbone

from .scripts.mistral_torch import ModelArgs
from .scripts.mistral_torch import Transformer as TorchTransformer

MODEL_PATH = pathlib.Path("mistral-7B-v0.1")


def port_weights(
    model_k3: MistralBackbone, model_torch: TorchTransformer, params: ModelArgs
):
    model_k3.get_layer("token_embedding").embeddings.assign(
        model_torch.tok_embeddings.weight.detach().cpu().numpy()
    )

    for i in range(model_k3.num_layers):
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.set_weights(
            [
                model_torch.layers[i]
                .attention.wk.weight.T.reshape(
                    params.dim, params.n_kv_heads, params.head_dim
                )
                .detach()
                .cpu()
                .numpy()
            ]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.set_weights(
            [
                model_torch.layers[i]
                .attention.wq.weight.T.reshape(
                    params.dim, params.n_heads, params.head_dim
                )
                .detach()
                .cpu()
                .numpy()
            ]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.set_weights(
            [
                model_torch.layers[i]
                .attention.wv.weight.T.reshape(
                    params.dim, params.n_kv_heads, params.head_dim
                )
                .detach()
                .cpu()
                .numpy()
            ]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.set_weights(
            [
                model_torch.layers[i]
                .attention.wo.weight.T.reshape(
                    params.n_heads, params.head_dim, params.dim
                )
                .detach()
                .cpu()
                .numpy()
            ]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.set_weights(
            [model_torch.layers[i].attention_norm.weight.detach().cpu().numpy()]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.set_weights(
            [
                model_torch.layers[i]
                .feed_forward.w3.weight.T.detach()
                .cpu()
                .numpy()
            ]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.set_weights(
            [
                model_torch.layers[i]
                .feed_forward.w2.weight.T.detach()
                .cpu()
                .numpy()
            ]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_gate_dense.set_weights(
            [
                model_torch.layers[i]
                .feed_forward.w1.weight.T.detach()
                .cpu()
                .numpy()
            ]
        )
        model_k3.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.set_weights(
            [model_torch.layers[i].ffn_norm.weight.detach().cpu().numpy()]
        )

    model_k3.get_layer("sequence_output_layernorm").set_weights(
        [model_torch.norm.weight.detach().cpu().numpy()]
    )
    model_k3.get_layer("token_embedding").reverse_embeddings.assign(
        model_torch.output.weight.T.detach().cpu().numpy()
    )


if __name__ == "__main__":
    with open(MODEL_PATH / "params.json", "r") as params_file:
        params = ModelArgs(**json.load(params_file))

    model_torch = TorchTransformer.from_folder(
        MODEL_PATH, device="cpu", dtype=torch.float16
    )
    print("Torch model loaded")
    model_k3 = MistralBackbone(
        vocabulary_size=32000,
        hidden_dim=4096,
        num_layers=32,
        num_query_heads=32,
        num_key_value_heads=8,
        intermediate_dim=14336,
        sliding_window=4096,
        layer_norm_epsilon=1e-6,
        dtype="float16",
    )
    print("Keras 3 model loaded.")

    port_weights(model_k3, model_torch, params)
    print("Weight transfer done.")

    model_k3.save_weights("mistral_7b.weights.h5")
    print("Weights saved.")
