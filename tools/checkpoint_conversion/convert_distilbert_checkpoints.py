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

import keras_nlp
from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    get_md5_checksum,
)

PRESET_MAP = {
    "distil_bert_base_en_uncased": "distilbert-base-uncased",
    "distil_bert_base_en_cased": "distilbert-base-cased",
    "distil_bert_base_multi_cased": "distilbert-base-multilingual-cased",
}

EXTRACT_DIR = "./{}"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def download_files(preset, hf_model_name):
    print("-> Download original vocab and config.")

    extract_dir = EXTRACT_DIR.format(preset)
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    # Config.
    config_path = os.path.join(extract_dir, "config.json")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/raw/main/config.json"
    )
    open(config_path, "wb").write(response.content)
    print(f"`{config_path}`")

    # Vocab.
    vocab_path = os.path.join(extract_dir, "vocab.txt")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/raw/main/vocab.txt"
    )
    open(vocab_path, "wb").write(response.content)
    print(f"`{vocab_path}`")


def define_preprocessor(preset, hf_model_name):
    print("\n-> Define the tokenizers.")
    extract_dir = EXTRACT_DIR.format(preset)
    vocab_path = os.path.join(extract_dir, "vocab.txt")

    keras_nlp_tokenizer = keras_nlp.models.DistilBertTokenizer(
        vocabulary=vocab_path,
    )
    keras_nlp_preprocessor = keras_nlp.models.DistilBertPreprocessor(
        keras_nlp_tokenizer
    )

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    print("\n-> Print MD5 checksum of the vocab files.")
    print(f"`{vocab_path}` md5sum: ", get_md5_checksum(vocab_path))

    return keras_nlp_preprocessor, hf_tokenizer


def convert_checkpoints(preset, keras_nlp_model, hf_model):
    print("\n-> Convert original weights to KerasNLP format.")

    extract_dir = EXTRACT_DIR.format(preset)
    config_path = os.path.join(extract_dir, "config.json")

    # Build config.
    cfg = {}
    with open(config_path, "r") as pt_cfg_handler:
        pt_cfg = json.load(pt_cfg_handler)
    cfg["vocabulary_size"] = pt_cfg["vocab_size"]
    cfg["num_layers"] = pt_cfg["n_layers"]
    cfg["num_heads"] = pt_cfg["n_heads"]
    cfg["hidden_dim"] = pt_cfg["dim"]
    cfg["intermediate_dim"] = pt_cfg["hidden_dim"]
    cfg["dropout"] = pt_cfg["dropout"]
    cfg["max_sequence_length"] = pt_cfg["max_position_embeddings"]

    print("Config:", cfg)

    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(
        str(hf_wts.keys())
        .replace(", ", "\n")
        .replace("odict_keys([", "")
        .replace("]", "")
        .replace(")", "")
    )

    keras_nlp_model.get_layer(
        "token_and_position_embedding"
    ).token_embedding.embeddings.assign(
        hf_wts["embeddings.word_embeddings.weight"]
    )
    keras_nlp_model.get_layer(
        "token_and_position_embedding"
    ).position_embedding.position_embeddings.assign(
        hf_wts["embeddings.position_embeddings.weight"]
    )

    keras_nlp_model.get_layer("embeddings_layer_norm").gamma.assign(
        hf_wts["embeddings.LayerNorm.weight"]
    )
    keras_nlp_model.get_layer("embeddings_layer_norm").beta.assign(
        hf_wts["embeddings.LayerNorm.bias"]
    )

    for i in range(keras_nlp_model.num_layers):
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            hf_wts[f"transformer.layer.{i}.attention.q_lin.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_wts[f"transformer.layer.{i}.attention.q_lin.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"transformer.layer.{i}.attention.k_lin.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_wts[f"transformer.layer.{i}.attention.k_lin.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"transformer.layer.{i}.attention.v_lin.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_wts[f"transformer.layer.{i}.attention.v_lin.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"transformer.layer.{i}.attention.out_lin.weight"]
            .transpose(1, 0)
            .reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_wts[f"transformer.layer.{i}.attention.out_lin.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            hf_wts[f"transformer.layer.{i}.sa_layer_norm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            hf_wts[f"transformer.layer.{i}.sa_layer_norm.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_wts[f"transformer.layer.{i}.ffn.lin1.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_wts[f"transformer.layer.{i}.ffn.lin1.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            hf_wts[f"transformer.layer.{i}.ffn.lin2.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            hf_wts[f"transformer.layer.{i}.ffn.lin2.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            hf_wts[f"transformer.layer.{i}.output_layer_norm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            hf_wts[f"transformer.layer.{i}.output_layer_norm.bias"].numpy()
        )

    # Save the model.
    print(f"\n-> Save KerasNLP model weights to `{preset}.h5`.")
    keras_nlp_model.save_weights(f"{preset}.h5")

    return keras_nlp_model


def check_output(
    preset,
    keras_nlp_preprocessor,
    keras_nlp_model,
    hf_tokenizer,
    hf_model,
):
    print("\n-> Check the outputs.")
    sample_text = ["cricket is awesome, easily the best sport in the world!"]

    # KerasNLP
    keras_nlp_inputs = keras_nlp_preprocessor(tf.constant(sample_text))
    keras_nlp_output = keras_nlp_model.predict(keras_nlp_inputs)

    # HF
    hf_inputs = hf_tokenizer(
        sample_text, padding="max_length", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasNLP output:", keras_nlp_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print("Difference:", np.mean(keras_nlp_output - hf_output.detach().numpy()))

    # Show the MD5 checksum of the model weights.
    print("Model md5sum: ", get_md5_checksum(f"./{preset}.h5"))


def main(_):
    hf_model_name = PRESET_MAP[FLAGS.preset]

    download_files(FLAGS.preset, hf_model_name)

    keras_nlp_preprocessor, hf_tokenizer = define_preprocessor(
        FLAGS.preset, hf_model_name
    )

    print("\n-> Load KerasNLP model.")
    keras_nlp_model = keras_nlp.models.DistilBertBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    print("\n-> Load HF model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    keras_nlp_model = convert_checkpoints(
        FLAGS.preset, keras_nlp_model, hf_model
    )

    check_output(
        FLAGS.preset,
        keras_nlp_preprocessor,
        keras_nlp_model,
        hf_tokenizer,
        hf_model,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
