# Copyright 2022 The KerasNLP Authors
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

import numpy as np
import requests
import tensorflow as tf
import transformers
from absl import app
from absl import flags

# Temporarily directly import gpt2 until we expose it.
from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    get_md5_checksum,
)

PRESET_MAP = {
    "gpt2_base": ("124M", "gpt2"),
    "gpt2_medium": ("355M", "gpt2-medium"),
    "gpt2_large": ("774M", "gpt2-large"),
    "gpt2_extra_large": ("1558M", "gpt2-xl"),
}

DOWNLOAD_SCRIPT_URL = (
    "https://raw.githubusercontent.com/openai/gpt-2/master/download_model.py"
)

EXTRACT_DIR = "./models/{}"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def download_model(preset, num_params):
    print("-> Download original weights.")
    response = requests.get(DOWNLOAD_SCRIPT_URL)
    open("download_model.py", "wb").write(response.content)

    os.system(f"python download_model.py {num_params}")


def convert_checkpoints(preset, num_params):
    print("\n-> Convert original weights to KerasNLP format.")
    # GPT-2 paths.
    extract_dir = EXTRACT_DIR.format(num_params)
    checkpoint_path = os.path.join(extract_dir, "model.ckpt")
    config_path = os.path.join(extract_dir, "hparams.json")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    print("Config:", cfg)

    print("Original weights:")
    vars = tf.train.list_variables(checkpoint_path)
    weights = {}
    for name, shape in vars:
        print(name, shape)
        weight = tf.train.load_variable(checkpoint_path, name)
        weights[name] = weight

    # Temporary direct import, as we aren't exposing this quite yet.
    keras_nlp_model = GPT2Backbone.from_preset(
        preset,
        load_weights=False,
    )

    keras_nlp_model.get_layer("token_embedding").embeddings.assign(
        weights["model/wte"]
    )
    keras_nlp_model.get_layer("position_embedding").position_embeddings.assign(
        weights["model/wpe"]
    )

    range_1 = (0, cfg["n_embd"])
    range_2 = (cfg["n_embd"], 2 * cfg["n_embd"])
    range_3 = (2 * cfg["n_embd"], 3 * cfg["n_embd"])

    for i in range(keras_nlp_model.num_layers):
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            weights[f"model/h{i}/attn/c_attn/w"][
                0, :, range_1[0] : range_1[1]
            ].reshape((cfg["n_embd"], cfg["n_head"], -1))
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            weights[f"model/h{i}/attn/c_attn/b"][
                range_1[0] : range_1[1]
            ].reshape((cfg["n_head"], -1))
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            weights[f"model/h{i}/attn/c_attn/w"][
                0, :, range_2[0] : range_2[1]
            ].reshape((cfg["n_embd"], cfg["n_head"], -1))
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            weights[f"model/h{i}/attn/c_attn/b"][
                range_2[0] : range_2[1]
            ].reshape((cfg["n_head"], -1))
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            weights[f"model/h{i}/attn/c_attn/w"][
                0, :, range_3[0] : range_3[1]
            ].reshape((cfg["n_embd"], cfg["n_head"], -1))
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            weights[f"model/h{i}/attn/c_attn/b"][
                range_3[0] : range_3[1]
            ].reshape((cfg["n_head"], -1))
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            weights[f"model/h{i}/attn/c_proj/w"][0].reshape(
                (cfg["n_head"], -1, cfg["n_embd"])
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            weights[f"model/h{i}/attn/c_proj/b"]
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.gamma.assign(weights[f"model/h{i}/ln_1/g"])
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.beta.assign(weights[f"model/h{i}/ln_1/b"])

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            weights[f"model/h{i}/mlp/c_fc/w"][0]
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            weights[f"model/h{i}/mlp/c_fc/b"]
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            weights[f"model/h{i}/mlp/c_proj/w"][0]
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            weights[f"model/h{i}/mlp/c_proj/b"]
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.gamma.assign(weights[f"model/h{i}/ln_2/g"])
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.beta.assign(weights[f"model/h{i}/ln_2/b"])

    keras_nlp_model.get_layer("layer_norm").gamma.assign(
        weights["model/ln_f/g"]
    )

    keras_nlp_model.get_layer("layer_norm").beta.assign(weights["model/ln_f/b"])

    # Save the model.
    print(f"\n-> Save KerasNLP model weights to `{preset}.h5`.")
    keras_nlp_model.save_weights(f"{preset}.h5")

    return keras_nlp_model


def define_tokenizer(preset, num_params, hf_model_name):
    print("\n-> Define the tokenizers.")
    extract_dir = extract_dir = EXTRACT_DIR.format(num_params)
    merges_path = os.path.join(extract_dir, "vocab.bpe")
    vocab_path = os.path.join(extract_dir, "encoder.json")

    keras_nlp_tokenizer = GPT2Tokenizer(
        vocabulary=vocab_path,
        merges=merges_path,
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    print("\n-> Print MD5 checksum of the vocab files.")
    print(f"`{vocab_path}` md5sum: ", get_md5_checksum(vocab_path))
    print(f"`{merges_path}` md5sum: ", get_md5_checksum(merges_path))

    return keras_nlp_tokenizer, hf_tokenizer


def check_output(
    preset,
    keras_nlp_model,
    keras_nlp_tokenizer,
    hf_model,
    hf_tokenizer,
):
    print("\n-> Check the outputs.")
    input_str = ["the quick brown fox ran, galloped and jumped."]

    # KerasNLP
    token_ids = keras_nlp_tokenizer(input_str)
    padding_mask = token_ids != 0

    keras_nlp_inputs = {
        "token_ids": token_ids.to_tensor(),
        "padding_mask": padding_mask.to_tensor(),
    }
    keras_nlp_output = keras_nlp_model.predict(keras_nlp_inputs)

    # HF
    hf_inputs = hf_tokenizer(input_str, return_tensors="pt")
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasNLP output:", keras_nlp_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print("Difference:", np.mean(keras_nlp_output - hf_output.detach().numpy()))

    # Show the MD5 checksum of the model weights.
    print("Model md5sum: ", get_md5_checksum(f"./{preset}.h5"))

    return keras_nlp_output


def main(_):
    assert (
        FLAGS.preset in PRESET_MAP.keys()
    ), f'Invalid preset {FLAGS.preset}. Must be one of {",".join(PRESET_MAP.keys())}'
    num_params = PRESET_MAP[FLAGS.preset][0]
    hf_model_name = PRESET_MAP[FLAGS.preset][1]

    download_model(FLAGS.preset, num_params)

    keras_nlp_model = convert_checkpoints(FLAGS.preset, num_params)

    print("\n-> Load HF model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    keras_nlp_tokenizer, hf_tokenizer = define_tokenizer(
        FLAGS.preset, num_params, hf_model_name
    )

    check_output(
        FLAGS.preset,
        keras_nlp_model,
        keras_nlp_tokenizer,
        hf_model,
        hf_tokenizer,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
