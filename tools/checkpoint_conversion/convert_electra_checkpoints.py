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
Electra weights conversion script.
"""

import json
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import huggingface_hub  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import transformers  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402

import keras_nlp  # noqa: E402
from keras_nlp.utils.preset_utils import save_to_preset  # noqa: E402

PRESET_MAP = {
    "electra_base_generator_en": "google/electra-base-generator",
    "electra_small_generator_en": "google/electra-small-generator",
    "electra_base_discriminator_en": "google/electra-base-discriminator",
    "electra_small_discriminator_en": "google/electra-small-discriminator",
    "electra_large_discriminator_en": "google/electra-large-discriminator",
    "electra_large_generator_en": "google/electra-large-generator",
}

EXTRACT_DIR = "./model"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    "electra_base_discriminator_en",
    f'Must be one of {",".join(PRESET_MAP)}',
)
flags.mark_flag_as_required("preset")


def download_hf_model(hf_model_name):
    hf_model_dir = huggingface_hub.snapshot_download(
        repo_id=hf_model_name,
        allow_patterns=["*.json", "*.bin"],
        ignore_patterns=["onx/*"],
        local_dir=EXTRACT_DIR,
    )
    return hf_model_dir


def convert_model(hf_model):
    hf_config = hf_model.config.to_dict()
    cfg = {}
    cfg["vocab_size"] = hf_config["vocab_size"]
    cfg["embedding_dim"] = hf_config["embedding_size"]
    cfg["num_layers"] = hf_config["num_hidden_layers"]
    cfg["num_heads"] = hf_config["num_attention_heads"]
    cfg["hidden_dim"] = hf_config["hidden_size"]
    cfg["intermediate_dim"] = hf_config["intermediate_size"]
    cfg["dropout"] = hf_config["hidden_dropout_prob"]
    cfg["max_sequence_length"] = hf_config["max_position_embeddings"]
    return keras_nlp.models.ElectraBackbone(**cfg)


def convert_tokenizer(hf_model_dir):
    tokenizer_path = os.path.join(hf_model_dir, "tokenizer.json")
    with open(tokenizer_path) as f:
        hf_tokenizer = json.load(f)
    vocab = hf_tokenizer["model"]["vocab"]

    return keras_nlp.models.ElectraTokenizer(vocabulary=vocab)


def convert_weights(keras_model, hf_model):
    hf_model_dict = hf_model.state_dict()

    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_model_dict["embeddings.word_embeddings.weight"].numpy()
    )
    keras_model.get_layer("position_embedding").position_embeddings.assign(
        hf_model_dict["embeddings.position_embeddings.weight"].numpy()
    )
    keras_model.get_layer("segment_embedding").embeddings.assign(
        hf_model_dict["embeddings.token_type_embeddings.weight"].numpy()
    )
    keras_model.get_layer("embeddings_layer_norm").gamma.assign(
        hf_model_dict["embeddings.LayerNorm.weight"]
    )
    keras_model.get_layer("embeddings_layer_norm").beta.assign(
        hf_model_dict["embeddings.LayerNorm.bias"]
    )

    if any(
        layer.name == "embeddings_projection" for layer in keras_model.layers
    ):
        keras_model.get_layer("embeddings_projection").kernel.assign(
            hf_model_dict["embeddings_project.weight"].transpose(1, 0).numpy()
        )
        keras_model.get_layer("embeddings_projection").bias.assign(
            hf_model_dict["embeddings_project.bias"]
        )

    for i in range(keras_model.num_layers):
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            hf_model_dict[f"encoder.layer.{i}.attention.self.query.weight"]
            .transpose(1, 0)
            .reshape((keras_model.hidden_dim, keras_model.num_heads, -1))
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_model_dict[f"encoder.layer.{i}.attention.self.query.bias"]
            .reshape((keras_model.num_heads, -1))
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_model_dict[f"encoder.layer.{i}.attention.self.key.weight"]
            .transpose(1, 0)
            .reshape((keras_model.hidden_dim, keras_model.num_heads, -1))
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_model_dict[f"encoder.layer.{i}.attention.self.key.bias"]
            .reshape((keras_model.num_heads, -1))
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_model_dict[f"encoder.layer.{i}.attention.self.value.weight"]
            .transpose(1, 0)
            .reshape((keras_model.hidden_dim, keras_model.num_heads, -1))
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_model_dict[f"encoder.layer.{i}.attention.self.value.bias"]
            .reshape((keras_model.num_heads, -1))
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_model_dict[f"encoder.layer.{i}.attention.output.dense.weight"]
            .transpose(1, 0)
            .reshape((keras_model.num_heads, -1, keras_model.hidden_dim))
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_model_dict[
                f"encoder.layer.{i}.attention.output.dense.bias"
            ].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer_norm.gamma.assign(
            hf_model_dict[
                f"encoder.layer.{i}.attention.output.LayerNorm.weight"
            ].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer_norm.beta.assign(
            hf_model_dict[
                f"encoder.layer.{i}.attention.output.LayerNorm.bias"
            ].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_model_dict[f"encoder.layer.{i}.intermediate.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_model_dict[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            hf_model_dict[f"encoder.layer.{i}.output.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            hf_model_dict[f"encoder.layer.{i}.output.dense.bias"].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layer_norm.gamma.assign(
            hf_model_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
        )
        keras_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layer_norm.beta.assign(
            hf_model_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
        )


def validate_output(keras_model, hf_model, keras_tokenizer, hf_tokenizer):
    input_str = ["The quick brown fox jumps over the lazy dog."]

    keras_nlp_preprocessor = keras_nlp.models.ElectraPreprocessor(
        keras_tokenizer
    )
    keras_nlp_inputs = keras_nlp_preprocessor(tf.constant(input_str))
    keras_nlp_output = keras_model.predict(keras_nlp_inputs).get(
        "sequence_output"
    )

    hf_inputs = hf_tokenizer(
        input_str, padding="max_length", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state.detach().numpy()
    print("🔶 KerasNLP output:", keras_nlp_output[0, 0, :10])
    print("🔶 HF output:", hf_output[0, 0, :10])
    print("Difference: ", np.mean(keras_nlp_output - hf_output))


def main(_):
    preset = FLAGS.preset
    assert preset in PRESET_MAP.keys(), f"Invalid preset: {preset}"
    print(f"✅ Converting {preset}")

    hf_model_name = PRESET_MAP[preset]
    hf_model_dir = download_hf_model(hf_model_name)
    print("✅ Downloaded model from Hugging face hub")

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    print(f"✅ Loaded {preset} from Hugging Face")

    keras_model = convert_model(hf_model)
    keras_tokenizer = convert_tokenizer(hf_model_dir)
    print("✅ Keras model loaded")

    convert_weights(keras_model, hf_model)
    print("✅ Weights converted")

    validate_output(keras_model, hf_model, keras_tokenizer, hf_tokenizer)
    print("✅ Validation complete")

    save_to_preset(keras_model, preset)
    save_to_preset(keras_tokenizer, preset, config_filename="tokenizer.json")

    print("✅ Preset saved")


if __name__ == "__main__":
    app.run(main)
