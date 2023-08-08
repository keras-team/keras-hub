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

import copy
import os

import h5py
import numpy as np
import requests
import tensorflow as tf
from transformers import TFXLNetModel
from transformers import XLNetTokenizer

from keras_nlp.models import XLNetBackbone

check_mems = False

PRESET = "xlnet-base-cased"
CKPT = f"https://huggingface.co/{PRESET}"
SAVE_PATH = "./tf_weights.h5"

# create HF model
hf_model = TFXLNetModel.from_pretrained(PRESET)

print(f"GPU Available or not : {tf.test.is_gpu_available()}")

with open(SAVE_PATH, "wb") as p:
    response = requests.get(CKPT + "/resolve/main/tf_model.h5")
    p.write(response.content)


tokenizer = XLNetTokenizer.from_pretrained(PRESET)
string = "An input text string."
tokens = tokenizer(string, return_tensors="tf", return_attention_mask=True)

tokenized_hf = copy.deepcopy(tokens)
tokenized_knlp = copy.deepcopy(tokens)

tokenized_knlp["token_ids"] = tokenized_knlp["input_ids"]
tokenized_knlp["padding_mask"] = tokenized_knlp["attention_mask"]
tokenized_knlp["segment_ids"] = tokenized_knlp["token_type_ids"]

del tokenized_knlp["attention_mask"]
del tokenized_knlp["input_ids"]
del tokenized_knlp["token_type_ids"]

# create keras_nlp model
knlp_model = XLNetBackbone(
    vocabulary_size=hf_model.config.vocab_size,
    num_layers=hf_model.config.n_layer,
    num_heads=hf_model.config.n_head,
    hidden_dim=hf_model.config.d_model,
    intermediate_dim=hf_model.config.d_inner,
    dropout=0.0,
    kernel_initializer_range=hf_model.config.initializer_range,
)
# Load weights for keras_nlp model
file_hf = h5py.File("./tf_weights.h5", "r")

try:
    _ = file_hf["transformer"]["tfxl_net_lm_head_model"]
    member = "tfxl_net_lm_head_model"
except:
    member = "tfxl_net_lm_head_model_1"


# Content and Query Embeddings

# mask emb
mask_emb = np.array(file_hf["transformer"][member]["transformer"]["mask_emb:0"])

# word emb
word_embed = np.array(
    file_hf["transformer"][member]["transformer"]["word_embedding"]["weight:0"]
)
knlp_model.get_layer("content_query_embedding").word_embed.embeddings.assign(
    word_embed
)
knlp_model.get_layer("encoder_block_attn_mask_layer").mask_emb.assign(mask_emb)

# Encoders
for i in range(hf_model.config.n_layer):
    # rel_attn
    # biases
    knlp_model.get_layer(f"xlnet_encoder_{i}").content_attention_bias.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["r_w_bias:0"]
        )
    )
    knlp_model.get_layer(f"xlnet_encoder_{i}").positional_attention_bias.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["r_r_bias:0"]
        )
    )
    knlp_model.get_layer(f"xlnet_encoder_{i}").segment_attention_bias.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["r_s_bias:0"]
        )
    )
    knlp_model.get_layer(f"xlnet_encoder_{i}").segment_encoding.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["seg_embed:0"]
        )
    )

    # layer-norm
    knlp_model.get_layer(f"xlnet_encoder_{i}").layer_norm.beta.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["layer_norm"]["beta:0"]
        )
    )
    knlp_model.get_layer(f"xlnet_encoder_{i}").layer_norm.gamma.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["layer_norm"]["gamma:0"]
        )
    )

    # rel_attn core
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).relative_attention._query_dense.kernel.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["q:0"]
        )
    )
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).relative_attention._key_dense.kernel.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["k:0"]
        )
    )
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).relative_attention._value_dense.kernel.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["v:0"]
        )
    )
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).relative_attention._output_dense.kernel.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["o:0"]
        )
    )
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).relative_attention._encoding_dense.kernel.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"][
                "rel_attn"
            ]["r:0"]
        )
    )

    # FF

    # FF layer 1
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).feedforward_intermediate_dense.kernel.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"]["ff"][
                "layer_1"
            ]["kernel:0"]
        )
    )
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).feedforward_intermediate_dense.bias.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"]["ff"][
                "layer_1"
            ]["bias:0"]
        )
    )

    # FF layer 2
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).feedforward_output_dense.kernel.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"]["ff"][
                "layer_2"
            ]["kernel:0"]
        )
    )
    knlp_model.get_layer(
        f"xlnet_encoder_{i}"
    ).feedforward_output_dense.bias.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"]["ff"][
                "layer_2"
            ]["bias:0"]
        )
    )

    # FF Layer Norm
    knlp_model.get_layer(f"xlnet_encoder_{i}").layer_norm_ff.beta.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"]["ff"][
                "layer_norm"
            ]["beta:0"]
        )
    )
    knlp_model.get_layer(f"xlnet_encoder_{i}").layer_norm_ff.gamma.assign(
        np.array(
            file_hf["transformer"][member]["transformer"][f"layer_._{i}"]["ff"][
                "layer_norm"
            ]["gamma:0"]
        )
    )

file_hf.close()

print("Model Weights Loaded!")

hf_preds = hf_model(tokenized_hf, training=False)
print(hf_preds["last_hidden_state"])

knlp_preds = knlp_model(tokenized_knlp, training=False)
print(knlp_preds, end="\n\n")

print(
    "Outputs matching or not for Last Hidden State : ",
    np.allclose(
        hf_preds["last_hidden_state"]
        .numpy()
        .reshape(-1, hf_model.config.d_model),
        knlp_preds.numpy().reshape(-1, hf_model.config.d_model),
        atol=1e-3,
    ),
)

# won't work since the recent version of the model doesn't return any mems!
if check_mems:
    for i in range(hf_model.config.n_layer):
        print(
            f"Outputs matching or not for Mem {i} : ",
            np.allclose(
                hf_preds["mems"][i]
                .numpy()
                .reshape(-1, hf_model.config.d_model),
                knlp_preds["new_mems"][i]
                .numpy()
                .reshape(-1, hf_model.config.d_model),
                atol=1e-3,
            ),
        )

os.remove(SAVE_PATH)
