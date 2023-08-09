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

import torch
from transformers import AutoModel

from keras_nlp.models.llama.llama_backbone import LlamaBackbone

os.environ["KERAS_BACKEND"] = "torch"

# from huggingface_hub import login
# llama weights as of now are on request access
# login(token='<your_huggingface_token')


PRESET_NAME = "Llama-2-7b-hf"
PRESET = "meta-llama/Llama-2-7b-hf"
EXTRACT_DIR = "./{}"


extract_dir = EXTRACT_DIR.format(PRESET_NAME)
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)


hf_model = AutoModel.from_pretrained(PRESET, use_auth_token=True)

hf_config = hf_model.config.to_dict()
hf_model.eval()
hf_wts = hf_model.state_dict()

cfg = {}

cfg["vocabulary_size"] = hf_config["vocab_size"]
cfg["num_layers"] = hf_config["num_hidden_layers"]
cfg["num_heads"] = hf_config["num_attention_heads"]
cfg["hidden_dim"] = hf_config["hidden_size"]
cfg["intermediate_dim"] = hf_config["intermediate_size"]
cfg["max_sequence_length"] = hf_config["max_position_embeddings"]
cfg["rope_scaling_type"] = hf_config["rope_scaling"]
cfg["layer_norm_epsilon"] = hf_config["rms_norm_eps"]
cfg["num_key_value_heads"] = hf_config["num_key_value_heads"]


keras_model = LlamaBackbone(**cfg)


keras_model.get_layer("token_embedding").embeddings.assign(
    hf_wts["embed_tokens.weight"]
)

for ilayer in range(cfg["num_layers"]):
    # attention layer
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._query_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.self_attn.q_proj.weight"]
        .numpy()
        .T.reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._key_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.self_attn.k_proj.weight"]
        .numpy()
        .T.reshape((cfg["hidden_dim"], cfg["num_key_value_heads"], -1))
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._value_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.self_attn.v_proj.weight"]
        .numpy()
        .T.reshape((cfg["hidden_dim"], cfg["num_key_value_heads"], -1))
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layer._output_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.self_attn.o_proj.weight"].numpy().T
    )

    # MLP
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_intermediate_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.mlp.up_proj.weight"].numpy().T
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_gate_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.mlp.gate_proj.weight"].numpy().T
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_output_dense.kernel.assign(
        hf_wts[f"layers.{ilayer}.mlp.down_proj.weight"].numpy().T
    )

    # LAYERNORM
    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._self_attention_layernorm.weight.assign(
        hf_wts[f"layers.{ilayer}.input_layernorm.weight"]
    )

    keras_model.get_layer(
        f"transformer_layer_{ilayer}"
    )._feedforward_layernorm.weight.assign(
        hf_wts[f"layers.{ilayer}.post_attention_layernorm.weight"]
    )


keras_model.get_layer("layer_norm").gamma.assign(hf_wts["norm.weight"])

token_ids = [1, 2181, 8522, 338]
padding_mask = [1, 1, 1, 1]

keras_inputs = {
    "token_ids": torch.tensor([token_ids]),
    "padding_mask": torch.tensor([padding_mask]),
}

with torch.no_grad():
    keras_outputs = keras_model(keras_inputs)
print("Keras output = ", keras_outputs.numpy())
