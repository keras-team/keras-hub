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
import tensorflow as tf
from transformers import AutoTokenizer
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

keras_model.get_layer("token_embedding").embeddings.assign(
    hf_model.embed_in.weight.detach().numpy()
)

for ilayer in range(cfg["num_layers"]):
    # attention layer
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._qkv_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.attention.query_key_value.weight"]
        .numpy()
        .T.reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._qkv_dense.bias.assign(
        hf_wts[f"layers.{ilayer}.attention.query_key_value.bias"].reshape(
            (cfg["num_heads"], -1)
        )
    )

    # Attention Dense
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._output_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.attention.dense.weight"].numpy().T
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._output_dense.bias.assign(
        hf_wts[f"layers.{ilayer}.attention.dense.bias"]
    )

    # LAYERNORM
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._input_layernorm.gamma.assign(
        hf_wts[f"layers.{ilayer}.input_layernorm.weight"]
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._input_layernorm.beta.assign(
        hf_wts[f"layers.{ilayer}.input_layernorm.bias"]
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layernorm.gamma.assign(
        hf_wts[f"layers.{ilayer}.post_attention_layernorm.weight"]
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layernorm.beta.assign(
        hf_wts[f"layers.{ilayer}.post_attention_layernorm.bias"]
    )

    # MLP
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_intermediate_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.mlp.dense_h_to_4h.weight"].numpy().T
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_intermediate_dense.bias.assign(
        hf_wts[f"layers.{ilayer}.mlp.dense_h_to_4h.bias"]
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_output_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.mlp.dense_4h_to_h.weight"].numpy().T
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_output_dense.bias.assign(
        hf_wts[f"layers.{ilayer}.mlp.dense_4h_to_h.bias"]
    )

    # Rotary Embedding
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer.rotary_embedding.inverse_freq.assign(
        hf_wts[f"layers.{ilayer}.attention.rotary_emb.inv_freq"]
    )

keras_model.get_layer("layer_norm").gamma.assign(
    hf_wts["final_layer_norm.weight"]
)

keras_model.get_layer("layer_norm").beta.assign(hf_wts["final_layer_norm.bias"])

hf_tokenizer = AutoTokenizer.from_pretrained(PRESET)
sample_text = ["cricket is awesome, easily the best sport in the world!"]
hf_inputs = hf_tokenizer(sample_text, return_tensors="pt")

keras_inputs = {
    "token_ids": tf.constant(
        [
            [
                68,
                4662,
                292,
                310,
                13103,
                13,
                4354,
                253,
                1682,
                9678,
                275,
                253,
                1533,
                2,
            ]
        ]
    ),
    "padding_mask": tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
}

keras_outputs = keras_model(keras_inputs)
print("Keras output = ", keras_outputs.numpy())
