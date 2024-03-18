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
"""
Falcon weight conversion script.

To run, install the CPU only development environment and huggingface libraries:
```
pip install -r requirements.txt
pip install transformers huggingface-cli
```

Login to Huggingface:
```
huggingface-cli login
```

Finally run this script to convert, validate and upload weights.
```
python tools/checkpoint_conversion/convert_falcon_checkpoints.py \
    --preset falcon_refinedweb_1b_en
```
"""

import json
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import absl  # noqa: E402
import huggingface_hub  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import keras_nlp  # noqa: E402

PRESET_MAP = {
    "falcon_refinedweb_1b_en": "tiiuae/falcon-rw-1b",
}

EXTRACT_DIR = "./model"

FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string(
    "preset",
    "falcon_refinedweb_1b_en",
    f'Must be one of {",".join(PRESET_MAP.keys())}.',
)


def download_hf_model(hf_model_name):
    hf_model_dir = huggingface_hub.snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=["*.json", "*.bin"],
        ignore_patterns=["onnx/*"],
        local_dir=EXTRACT_DIR,
    )

    return hf_model_dir


def convert_model(hf_model):
    hf_config = hf_model.config.to_dict()
    kwargs = {}
    kwargs["vocabulary_size"] = hf_config["vocab_size"]
    kwargs["num_layers"] = hf_config["num_hidden_layers"]
    kwargs["num_attention_heads"] = hf_config["num_attention_heads"]
    kwargs["hidden_dim"] = hf_config["hidden_size"]
    kwargs["intermediate_dim"] = 4 * kwargs["hidden_dim"]
    kwargs["feedforward_dropout_rate"] = hf_config["hidden_dropout"]
    kwargs["attention_dropout_rate"] = hf_config["attention_dropout"]

    return keras_nlp.models.FalconBackbone(**kwargs)


def convert_tokenizer(hf_model_dir):
    tokenizer_file_path = os.path.join(hf_model_dir, "tokenizer.json")
    with open(tokenizer_file_path) as tokenizer_file:
        hf_tokenizer = json.load(tokenizer_file)

    vocab = hf_tokenizer["model"]["vocab"]
    merges = hf_tokenizer["model"]["merges"]
    return keras_nlp.models.FalconTokenizer(vocabulary=vocab, merges=merges)


def convert_weights(keras_model, hf_model):
    hf_model.eval()
    hf_wts = hf_model.state_dict()

    # token_embedding.
    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["word_embeddings.weight"]
    )

    for ilayer in range(keras_model.num_layers):
        # Split key query value.
        fused_qkv = (
            hf_wts[f"h.{ilayer}.self_attention.query_key_value.weight"]
            .numpy()
            .T
        )
        seq_length, _ = fused_qkv.shape
        head_dim = keras_model.hidden_dim // keras_model.num_attention_heads
        fused_qkv = fused_qkv.reshape(
            seq_length, keras_model.num_attention_heads, 3, head_dim
        )
        query, key, value = (
            fused_qkv[..., 0, :],
            fused_qkv[..., 1, :],
            fused_qkv[..., 2, :],
        )

        fused_bias = hf_wts[
            f"h.{ilayer}.self_attention.query_key_value.bias"
        ].numpy()
        fused_bias = fused_bias.reshape(
            keras_model.num_attention_heads, 3, head_dim
        )
        query_bias, key_bias, value_bias = (
            fused_bias[..., 0, :],
            fused_bias[..., 1, :],
            fused_bias[..., 2, :],
        )

        # TODO: check if bias is true before assigning bias.
        # Attention/query.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.query_dense.kernel.assign(query)
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.query_dense.bias.assign(query_bias)

        # Attention/key.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.key_dense.kernel.assign(key)
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.key_dense.bias.assign(key_bias)

        # Attention/value.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.value_dense.kernel.assign(value)
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.value_dense.bias.assign(value_bias)

        # Attention/dense.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.output_dense.kernel.assign(
            hf_wts[f"h.{ilayer}.self_attention.dense.weight"].T.numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).attention_layer.output_dense.bias.assign(
            hf_wts[f"h.{ilayer}.self_attention.dense.bias"].numpy()
        )

        # MLP/dense_h_to_4h.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).dense_h_to_4h.kernel.assign(
            hf_wts[f"h.{ilayer}.mlp.dense_h_to_4h.weight"].T.numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).dense_h_to_4h.bias.assign(
            hf_wts[f"h.{ilayer}.mlp.dense_h_to_4h.bias"].numpy()
        )

        # MLP/dense_4h_to_h.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).dense_4h_to_h.kernel.assign(
            hf_wts[f"h.{ilayer}.mlp.dense_4h_to_h.weight"].T.numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).dense_4h_to_h.bias.assign(
            hf_wts[f"h.{ilayer}.mlp.dense_4h_to_h.bias"].numpy()
        )

        # input_layernorm.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).input_layernorm.gamma.assign(
            hf_wts[f"h.{ilayer}.input_layernorm.weight"]
        )
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).input_layernorm.beta.assign(
            hf_wts[f"h.{ilayer}.input_layernorm.bias"]
        )

        # post_attention_layernorm.
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).post_attention_layernorm.gamma.assign(
            hf_wts[f"h.{ilayer}.post_attention_layernorm.weight"].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        ).post_attention_layernorm.beta.assign(
            hf_wts[f"h.{ilayer}.post_attention_layernorm.bias"].numpy()
        )

    # final_layernorm.
    keras_model.get_layer("final_layernorm").gamma.assign(
        hf_wts["ln_f.weight"].numpy()
    )
    keras_model.get_layer("final_layernorm").beta.assign(
        hf_wts["ln_f.bias"].numpy()
    )


def validate_output(
    hf_model,
    keras_model,
    hf_tokenizer,
    keras_tokenizer,
):
    input_str = ["the quick brown fox ran, galloped and jumped."]

    # KerasNLP model.
    token_ids = torch.tensor(keras_tokenizer(input_str))
    padding_mask = token_ids != 3
    keras_model_input = {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }
    keras_model_outputs = keras_model.predict(keras_model_input)

    # HuggingFace model.
    hf_model_input = hf_tokenizer(input_str, return_tensors="pt")

    activation = {}

    def get_activation(name):
        def hook(hf_model, input, output):
            activation[name] = output[0].detach()

        return hook

    hf_model.register_forward_hook(get_activation("ln_f"))
    hf_model(**hf_model_input)
    hf_model_outputs = activation["ln_f"].detach().numpy()

    # Comparing the outputs.
    print("🔶 KerasNLP tokens ids:", keras_model_input["token_ids"])
    print("🔶 HF tokens ids:", hf_model_input["input_ids"])
    print("🔶 KerasNLP output:", keras_model_outputs[0, 1, :10])
    print("🔶 HF output:", hf_model_outputs[0, 1, :10])
    print("🔶 Difference:", np.mean(keras_model_outputs - hf_model_outputs))


def main(_):
    preset = FLAGS.preset
    print(f"✅ Coverting {preset}")

    hf_model_name = PRESET_MAP[preset]
    hf_model_dir = download_hf_model(hf_model_name)
    print("✅ Huggingface model downloaded from hub")

    hf_model = transformers.FalconModel.from_pretrained(hf_model_dir)
    # Falcon uses GPT2 tokenizer.
    hf_tokenizer = transformers.GPT2TokenizerFast.from_pretrained(hf_model_dir)
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
    absl.app.run(main)
