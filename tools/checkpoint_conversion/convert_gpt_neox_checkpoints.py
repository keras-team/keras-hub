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
import os

import requests
from transformers import GPTNeoXModel

from keras_nlp.models.gpt_neox.gpt_neox_backbone import GPTNeoXBackbone

PRESET_NAME = "pythia-70m"
PRESET = "EleutherAI/pythia-70m-deduped"
EXTRACT_DIR = "./{}"


extract_dir = EXTRACT_DIR.format(PRESET_NAME)
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Config.
config_path = os.path.join(extract_dir, "config.json")
response = requests.get(f"https://huggingface.co/{PRESET}/raw/main/config.json")
open(config_path, "wb").write(response.content)

# Vocab.
spm_path = os.path.join(extract_dir, "spm.model")
response = requests.get(
    f"https://huggingface.co/{PRESET}/resolve/main/spm.model"
)
open(spm_path, "wb").write(response.content)

cfg = {}
with open(config_path, "r") as pt_cfg_handler:
    pt_cfg = json.load(pt_cfg_handler)

cfg["vocabulary_size"] = pt_cfg["vocab_size"]
cfg["num_layers"] = pt_cfg["num_hidden_layers"]
cfg["num_heads"] = pt_cfg["num_attention_heads"]
cfg["hidden_dim"] = pt_cfg["hidden_size"]
cfg["intermediate_dim"] = pt_cfg["intermediate_size"]
cfg["max_sequence_length"] = pt_cfg["max_position_embeddings"]
cfg["layer_norm_epsilon"] = pt_cfg["layer_norm_eps"]
cfg["rotary_pct"] = pt_cfg["rotary_pct"]
cfg["rotary_emb_dim"] = pt_cfg["rotary_emb_dim"]

hf_model = GPTNeoXModel.from_pretrained(PRESET)
hf_model.eval()

hf_wts = hf_model.state_dict()

keras_model = GPTNeoXBackbone(**cfg)

keras_model.get_layer("token_embedding").embeddings.assign(
    hf_model.embed_in.weight.detach().numpy()
)

for i in range(cfg["num_layers"]):

    # attention layer

    # QUERY
    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._query_dense.kernel.assign(
        hf_wts[f"layers.{i}.attention.query_key_value.weight"][
            0 : cfg["hidden_dim"], :
        ].reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._query_dense.bias.assign(
        hf_wts[f"layers.{i}.attention.query_key_value.bias"][
            0 : cfg["hidden_dim"]
        ].reshape((cfg["num_heads"], -1))
    )

    # KEY
    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._key_dense.kernel.assign(
        hf_wts[f"layers.{i}.attention.query_key_value.weight"][
            cfg["hidden_dim"] : 2 * cfg["hidden_dim"], :
        ].reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._key_dense.bias.assign(
        hf_wts[f"layers.{i}.attention.query_key_value.bias"][
            cfg["hidden_dim"] : 2 * cfg["hidden_dim"]
        ].reshape((cfg["num_heads"], -1))
    )

    # VALUE
    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._value_dense.kernel.assign(
        hf_wts[f"layers.{i}.attention.query_key_value.weight"][
            2 * cfg["hidden_dim"] : 3 * cfg["hidden_dim"], :
        ].reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._value_dense.bias.assign(
        hf_wts[f"layers.{i}.attention.query_key_value.bias"][
            2 * cfg["hidden_dim"] : 3 * cfg["hidden_dim"]
        ].reshape((cfg["num_heads"], -1))
    )

    # Attention Dense
    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._output_dense.kernel.assign(
        hf_wts[f"layers.{i}.attention.dense.weight"]
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer._output_dense.bias.assign(
        hf_wts[f"layers.{i}.attention.dense.bias"]
    )

    # LAYERNORM
    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layernorm.gamma.assign(
        hf_wts[f"layers.{i}.input_layernorm.weight"]
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layernorm.beta.assign(
        hf_wts[f"layers.{i}.input_layernorm.bias"]
    )

    # MLP
    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._feedforward_intermediate_dense.kernel.assign(
        hf_wts[f"layers.{i}.mlp.dense_h_to_4h.weight"].numpy().T
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._feedforward_intermediate_dense.bias.assign(
        hf_wts[f"layers.{i}.mlp.dense_h_to_4h.bias"]
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._feedforward_output_dense.kernel.assign(
        hf_wts[f"layers.{i}.mlp.dense_4h_to_h.weight"].numpy().T
    )

    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._feedforward_output_dense.bias.assign(
        hf_wts[f"layers.{i}.mlp.dense_4h_to_h.bias"]
    )

    # Rotary Embedding
    keras_model.get_layer(
        f"transformer_layer_{i}"
    )._self_attention_layer.rotary_embedding.inverse_freq.assign(
        hf_wts[f"layers.{i}.attention.rotary_emb.inv_freq"]
    )

keras_model.get_layer("layer_norm").gamma.assign(
    hf_wts["final_layer_norm.weight"]
)

keras_model.get_layer("layer_norm").beta.assign(hf_wts["final_layer_norm.bias"])
