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

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from checkpoint_conversion_utils import get_md5_checksum  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

import keras_nlp  # noqa: E402
from keras_nlp.models import BloomBackbone  # noqa: E402
from keras_nlp.models import BloomTokenizer  # noqa: E402

FLAGS = flags.FLAGS

PRESET_MAP = {
    "bloom_tiny": "bigscience/bloom-560m",
    "bloom_extra_small": "bigscience/bloom-1b1",
    "bloom_small": "bigscience/bloom-1b7",
    "bloom_meduim": "bigscience/bloom-3b",
    "bloom_large": "bigscience/bloom-7b1",
    "bloom_extra_large": "bigscience/bloom",
}

EXTRACT_DIR = "./model"


flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)
flags.mark_flag_as_required("preset")


def download_hf_model(hf_model_name):
    hf_model_dir = snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=["*.json", "*.bin"],
        ignore_patterns=["onnx/*"],
        local_dir=EXTRACT_DIR,
    )

    return hf_model_dir


def convert_model(hf_model):
    # get huggingface model configuration.
    hf_config = hf_model.config.to_dict()

    kwargs = {}
    kwargs["vocabulary_size"] = hf_config["vocab_size"]
    kwargs["num_layers"] = hf_config["n_layer"]
    kwargs["num_heads"] = hf_config["n_head"]
    kwargs["hidden_dim"] = hf_config["hidden_size"]
    kwargs["intermediate_dim"] = hf_config["hidden_size"] * 4
    kwargs["dropout"] = hf_config["hidden_dropout"]
    kwargs["layer_norm_epsilon"] = hf_config["layer_norm_epsilon"]

    return BloomBackbone(**kwargs)


def convert_tokenizer(hf_model_dir):
    tokenizer_file_path = os.path.join(hf_model_dir, "tokenizer.json")
    with open(tokenizer_file_path) as tokenizer_file:
        hf_tokenizer = json.load(tokenizer_file)

    vocab = hf_tokenizer["model"]["vocab"]
    merges = hf_tokenizer["model"]["merges"]

    return BloomTokenizer(vocabulary=vocab, merges=merges)


def convert_weights(keras_model, hf_model):
    hidden_dim = keras_model.hidden_dim
    num_heads = keras_model.num_heads
    head_dim = hidden_dim // num_heads
    num_layers = keras_model.num_layers

    # get huggingface model weights.
    hf_wts = hf_model.state_dict()

    # assign huggingface weights to the keras model.
    # Embedding layer.
    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["word_embeddings.weight"]
    )
    # LayerNorm.
    keras_model.get_layer("token_embedding_layernorm").gamma.assign(
        hf_wts["word_embeddings_layernorm.weight"]
    )
    keras_model.get_layer("token_embedding_layernorm").beta.assign(
        hf_wts["word_embeddings_layernorm.bias"]
    )

    keras_model.get_layer("final_layernorm").gamma.assign(hf_wts["ln_f.weight"])
    keras_model.get_layer("final_layernorm").beta.assign(hf_wts["ln_f.bias"])

    # Decoder layers.
    for i in range(num_layers):
        decoder_layer = keras_model.get_layer(f"transformer_layer_{i}")
        # LayrNorm.
        decoder_layer._pre_attention_layernorm.gamma.assign(
            hf_wts[f"h.{i}.input_layernorm.weight"]
        )
        decoder_layer._pre_attention_layernorm.beta.assign(
            hf_wts[f"h.{i}.input_layernorm.bias"]
        )
        decoder_layer._post_attention_layernorm.gamma.assign(
            hf_wts[f"h.{i}.post_attention_layernorm.weight"]
        )
        decoder_layer._post_attention_layernorm.beta.assign(
            hf_wts[f"h.{i}.post_attention_layernorm.bias"]
        )

        # Attention layer.
        attention_layer = decoder_layer._self_attention_layer

        fused_qkv_kernal = hf_wts[
            f"h.{i}.self_attention.query_key_value.weight"
        ].T
        fused_qkv_kernal = fused_qkv_kernal.view(
            hidden_dim, num_heads, 3, head_dim
        )
        query_kernal = fused_qkv_kernal[..., 0, :]
        key_kernal = fused_qkv_kernal[..., 1, :]
        value_kernl = fused_qkv_kernal[..., 2, :]

        fused_qkv_bais = hf_wts[f"h.{i}.self_attention.query_key_value.bias"]
        fused_qkv_bais = fused_qkv_bais.view(num_heads, 3, head_dim)
        query_bais = fused_qkv_bais[:, 0, :]
        key_bais = fused_qkv_bais[:, 1, :]
        value_bais = fused_qkv_bais[:, 2, :]

        attention_layer._query_dense.kernel.assign(query_kernal)
        attention_layer._query_dense.bias.assign(query_bais)
        attention_layer._key_dense.kernel.assign(key_kernal)
        attention_layer._key_dense.bias.assign(key_bais)
        attention_layer._value_dense.kernel.assign(value_kernl)
        attention_layer._value_dense.bias.assign(value_bais)

        attention_layer._output_dense.kernel.assign(
            hf_wts[f"h.{i}.self_attention.dense.weight"].T
        )
        attention_layer._output_dense.bias.assign(
            hf_wts[f"h.{i}.self_attention.dense.bias"]
        )

        # mlp.
        decoder_layer._mlp_intermediate_dense.kernel.assign(
            hf_wts[f"h.{i}.mlp.dense_h_to_4h.weight"].T
        )
        decoder_layer._mlp_intermediate_dense.bias.assign(
            hf_wts[f"h.{i}.mlp.dense_h_to_4h.bias"]
        )
        decoder_layer._mlp_output_dense.kernel.assign(
            hf_wts[f"h.{i}.mlp.dense_4h_to_h.weight"].T
        )
        decoder_layer._mlp_output_dense.bias.assign(
            hf_wts[f"h.{i}.mlp.dense_4h_to_h.bias"]
        )

def validate_output(
    hf_model,
    keras_model,
    hf_tokenizer,
    keras_tokenizer,
):
    print("⏺ checking Backbone output")

    hf_model_input = {
        "input_ids": torch.tensor([[59414, 15, 2670, 35433, 632, 207595]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
    }

    hf_model_outputs = hf_model(**hf_model_input)
    hf_model_outputs = hf_model_outputs.last_hidden_state
    hf_model_outputs = hf_model_outputs.detach().numpy()

    keras_model_input = {
        "token_ids": torch.tensor([[59414, 15, 2670, 35433, 632, 207595]]),
        "padding_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
    }

    keras_model_outputs = keras_model.predict(keras_model_input)

    # Comparing the outputs.
    print("🔶 KerasNLP output:", keras_model_outputs[0, 0, :10])
    print("🔶 HF output:", hf_model_outputs[0, 0, :10])
    print("🔶 Difference:", np.mean(keras_model_outputs - hf_model_outputs))

    print("⏺ checking tokenizer output")

    input_str = ["the quick brown fox ran, galloped and jumped."]
    token_ids_keras = keras_tokenizer(input_str)
    token_ids_hf = hf_tokenizer(input_str)

    print("🔶 KerasNLP output:", token_ids_keras)
    print("🔶 HF output:", token_ids_hf)


def main(_):
    preset = FLAGS.preset

    assert (
        preset in PRESET_MAP.keys()
    ), f'Invalid preset {preset}. Must be one of {",".join(PRESET_MAP.keys())}'

    print(f"✅ Coverting {preset}")

    hf_model_name = PRESET_MAP[preset]
    hf_model_dir = download_hf_model(hf_model_name)
    print("✅ Huggingface model downloaded from hub")

    hf_model = transformers.BloomModel.from_pretrained(hf_model_dir)
    hf_tokenizer = transformers.BloomTokenizerFast.from_pretrained(hf_model_dir)
    print("✅ Huggingface model loaded")

    keras_model = convert_model(hf_model)
    keras_tokenizer = convert_tokenizer(hf_model_dir)
    print("✅ Keras model loaded")

    convert_weights(keras_model, hf_model)
    print("✅ Weights converted")

    validate_output(
        hf_model,
        keras_model,
        hf_tokenizer,
        keras_tokenizer,
    )
    print("✅ Numerics validated")

    keras_nlp.src.utils.preset_utils.save_to_preset(keras_model, preset)
    keras_nlp.src.utils.preset_utils.save_to_preset(
        keras_tokenizer, preset, config_filename="tokenizer.json"
    )
    print("✅ Preset saved")


if __name__ == "__main__":
    app.run(main)
