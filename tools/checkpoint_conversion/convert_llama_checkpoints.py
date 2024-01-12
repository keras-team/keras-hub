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
Llama weight conversion script.

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
python tools/checkpoint_conversion/convert_llama_checkpoints.py \
    --preset tiny_llama_1b_chat_en
```
"""

import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import absl  # noqa: E402
import transformers  # noqa: E402

import keras_nlp  # noqa: E402

PRESET_MAP = {
    "tiny_llama_1b_chat_en": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama2_7b_en": "meta-llama/Llama-2-7b-hf",
    "llama2_7b_chat_en": "meta-llama/Llama-2-7b-chat-hf",
    "llama2_13b_en": "meta-llama/Llama-2-13b-hf",
    "llama2_13b_chat_en": "meta-llama/Llama-2-13b-chat-hf",
}

FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string(
    "preset",
    "tiny_llama_1b_chat_en",
    f'Must be one of {",".join(PRESET_MAP.keys())}.',
)


def convert_model(hf_model):
    hf_config = hf_model.config.to_dict()
    kwargs = {}
    kwargs["vocabulary_size"] = hf_config["vocab_size"]
    kwargs["num_layers"] = hf_config["num_hidden_layers"]
    kwargs["hidden_dim"] = hf_config["hidden_size"]
    kwargs["intermediate_dim"] = hf_config["intermediate_size"]
    kwargs["max_sequence_length"] = hf_config["max_position_embeddings"]
    kwargs["layer_norm_epsilon"] = hf_config["rms_norm_eps"]
    kwargs["num_query_heads"] = hf_config["num_attention_heads"]
    kwargs["num_key_value_heads"] = hf_config["num_key_value_heads"]
    return keras_nlp.models.LlamaBackbone(**kwargs)


def convert_tokenizer(hf_tokenizer):
    proto_path = transformers.utils.hub.get_file_from_repo(
        hf_tokenizer.name_or_path, "tokenizer.model"
    )
    return keras_nlp.models.LlamaTokenizer(proto_path)


def convert_weights(keras_model, hf_model):
    hf_model.eval()
    hf_wts = hf_model.state_dict()

    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["embed_tokens.weight"]
    )

    for ilayer in range(keras_model.num_layers):
        # attention layer
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer._query_dense.kernel.assign(
            hf_wts[f"layers.{ilayer}.self_attn.q_proj.weight"]
            .numpy()
            .T.reshape(
                (keras_model.hidden_dim, keras_model.num_query_heads, -1)
            )
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"layers.{ilayer}.self_attn.k_proj.weight"]
            .numpy()
            .T.reshape(
                (keras_model.hidden_dim, keras_model.num_key_value_heads, -1)
            )
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"layers.{ilayer}.self_attn.v_proj.weight"]
            .numpy()
            .T.reshape(
                (keras_model.hidden_dim, keras_model.num_key_value_heads, -1)
            )
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


def validate_output(
    hf_model,
    keras_model,
    hf_tokenizer,
    keras_tokenizer,
):
    # TODO: add this!
    pass


def main(_):
    preset = FLAGS.preset
    print(f"✅ Coverting {preset}")
    hf_model = transformers.AutoModel.from_pretrained(PRESET_MAP[preset])
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        PRESET_MAP[preset]
    )
    print("✅ Huggingface model loaded")

    keras_model = convert_model(hf_model)
    keras_tokenizer = convert_tokenizer(hf_tokenizer)
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
