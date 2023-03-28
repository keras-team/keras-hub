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

import numpy as np
import requests
import tensorflow as tf
import transformers
from absl import app
from absl import flags
from checkpoint_conversion_utils import get_md5_checksum

from keras_nlp.models.deberta_v3.deberta_v3_backbone import DebertaV3Backbone
from keras_nlp.models.deberta_v3.deberta_v3_preprocessor import (
    DebertaV3Preprocessor,
)
from keras_nlp.models.deberta_v3.deberta_v3_tokenizer import DebertaV3Tokenizer

PRESET_MAP = {
    "deberta_v3_extra_small_en": "microsoft/deberta-v3-xsmall",
    "deberta_v3_small_en": "microsoft/deberta-v3-small",
    "deberta_v3_base_en": "microsoft/deberta-v3-base",
    "deberta_v3_large_en": "microsoft/deberta-v3-large",
    "deberta_v3_base_multi": "microsoft/mdeberta-v3-base",
}

EXTRACT_DIR = "./{}"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def download_files(hf_model_name):
    print("-> Download original vocabulary and config.")

    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
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
    spm_path = os.path.join(extract_dir, "spm.model")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/resolve/main/spm.model"
    )
    open(spm_path, "wb").write(response.content)
    print(f"`{spm_path}`")


def define_preprocessor(hf_model_name):
    print("\n-> Define the tokenizers.")
    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    spm_path = os.path.join(extract_dir, "spm.model")

    keras_nlp_tokenizer = DebertaV3Tokenizer(proto=spm_path)

    # Avoid having padding tokens. This is because the representations of the
    # padding token may be vastly different from the representations computed in
    # the original model. See https://github.com/keras-team/keras/pull/16619#issuecomment-1156338394.
    sequence_length = 14
    if FLAGS.preset == "deberta_v3_base_multi":
        sequence_length = 17
    keras_nlp_preprocessor = DebertaV3Preprocessor(
        keras_nlp_tokenizer, sequence_length=sequence_length
    )

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    print("\n-> Print MD5 checksum of the vocab files.")
    print(f"`{spm_path}` md5sum: ", get_md5_checksum(spm_path))

    return keras_nlp_preprocessor, hf_tokenizer


def convert_checkpoints(keras_nlp_model, hf_model):
    print("\n-> Convert original weights to KerasNLP format.")

    extract_dir = EXTRACT_DIR.format(FLAGS.preset)
    config_path = os.path.join(extract_dir, "config.json")

    # Build config.
    cfg = {}
    with open(config_path, "r") as pt_cfg_handler:
        pt_cfg = json.load(pt_cfg_handler)
    cfg["vocabulary_size"] = pt_cfg["vocab_size"]
    cfg["num_layers"] = pt_cfg["num_hidden_layers"]
    cfg["num_heads"] = pt_cfg["num_attention_heads"]
    cfg["hidden_dim"] = pt_cfg["hidden_size"]
    cfg["intermediate_dim"] = pt_cfg["intermediate_size"]
    cfg["dropout"] = pt_cfg["hidden_dropout_prob"]
    cfg["max_sequence_length"] = pt_cfg["max_position_embeddings"]
    cfg["bucket_size"] = pt_cfg["position_buckets"]
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

    keras_nlp_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["embeddings.word_embeddings.weight"]
    )
    keras_nlp_model.get_layer("embeddings_layer_norm").gamma.assign(
        hf_wts["embeddings.LayerNorm.weight"]
    )
    keras_nlp_model.get_layer("embeddings_layer_norm").beta.assign(
        hf_wts["embeddings.LayerNorm.bias"]
    )
    keras_nlp_model.get_layer("rel_embedding").rel_embeddings.assign(
        hf_wts["encoder.rel_embeddings.weight"]
    )
    keras_nlp_model.get_layer("rel_embedding").layer_norm.gamma.assign(
        hf_wts["encoder.LayerNorm.weight"]
    )
    keras_nlp_model.get_layer("rel_embedding").layer_norm.beta.assign(
        hf_wts["encoder.LayerNorm.bias"]
    )

    for i in range(keras_nlp_model.num_layers):
        # Q,K,V
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.query_proj.weight"]
            .numpy()
            .T.reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.query_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.key_proj.weight"]
            .numpy()
            .T.reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.key_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.value_proj.weight"]
            .numpy()
            .T.reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.self.value_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        # Attn output.
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.attention.output.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.attention.output.dense.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            hf_wts[
                f"encoder.layer.{i}.attention.output.LayerNorm.weight"
            ].numpy()
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            hf_wts[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy()
        )

        # Intermediate FF layer.
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
        )

        # Output FF layer.
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.weight"].numpy().T
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"disentangled_attention_encoder_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
        )

    # Save the model.
    print(f"\n-> Save KerasNLP model weights to `{FLAGS.preset}.h5`.")
    keras_nlp_model.save_weights(f"{FLAGS.preset}.h5")

    return keras_nlp_model


def check_output(
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
        sample_text, padding="longest", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasNLP output:", keras_nlp_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print("Difference:", np.mean(keras_nlp_output - hf_output.detach().numpy()))

    # Show the MD5 checksum of the model weights.
    print("Model md5sum: ", get_md5_checksum(f"./{FLAGS.preset}.h5"))


def main(_):
    hf_model_name = PRESET_MAP[FLAGS.preset]

    download_files(hf_model_name)

    keras_nlp_preprocessor, hf_tokenizer = define_preprocessor(hf_model_name)

    print("\n-> Load KerasNLP model.")
    keras_nlp_model = DebertaV3Backbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    print("\n-> Load HF model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    keras_nlp_model = convert_checkpoints(keras_nlp_model, hf_model)

    check_output(
        keras_nlp_preprocessor,
        keras_nlp_model,
        hf_tokenizer,
        hf_model,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
