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
import shutil

import numpy as np
import tensorflow as tf
import transformers
from absl import app
from absl import flags
from checkpoint_conversion_utils import get_md5_checksum

import keras_nlp

PRESET_MAP = {
    "f_net_base_en": "google/fnet-base",
    "f_net_large_en": "google/fnet-large",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def convert_checkpoints(hf_model):
    print("\n-> Convert original weights to KerasNLP format.")

    print("\n-> Load KerasNLP model.")
    keras_nlp_model = keras_nlp.models.FNetBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(list(hf_wts.keys()))

    keras_nlp_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["embeddings.word_embeddings.weight"]
    )
    keras_nlp_model.get_layer("position_embedding").position_embeddings.assign(
        hf_wts["embeddings.position_embeddings.weight"]
    )
    keras_nlp_model.get_layer("segment_embedding").embeddings.assign(
        hf_wts["embeddings.token_type_embeddings.weight"]
    )

    keras_nlp_model.get_layer("embeddings_layer_norm").gamma.assign(
        hf_wts["embeddings.LayerNorm.weight"]
    )
    keras_nlp_model.get_layer("embeddings_layer_norm").beta.assign(
        hf_wts["embeddings.LayerNorm.bias"]
    )

    keras_nlp_model.get_layer("embedding_projection").kernel.assign(
        hf_wts["embeddings.projection.weight"].T
    )
    keras_nlp_model.get_layer("embedding_projection").bias.assign(
        hf_wts["embeddings.projection.bias"]
    )

    for i in range(keras_nlp_model.num_layers):
        keras_nlp_model.get_layer(
            f"f_net_layer_{i}"
        )._mixing_layer_norm.gamma.assign(
            hf_wts[f"encoder.layer.{i}.fourier.output.LayerNorm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"f_net_layer_{i}"
        )._mixing_layer_norm.beta.assign(
            hf_wts[f"encoder.layer.{i}.fourier.output.LayerNorm.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"f_net_layer_{i}"
        )._intermediate_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"f_net_layer_{i}"
        )._intermediate_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"f_net_layer_{i}"
        )._output_dense.kernel.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.weight"]
            .transpose(1, 0)
            .numpy()
        )
        keras_nlp_model.get_layer(f"f_net_layer_{i}")._output_dense.bias.assign(
            hf_wts[f"encoder.layer.{i}.output.dense.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"f_net_layer_{i}"
        )._output_layer_norm.gamma.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"f_net_layer_{i}"
        )._output_layer_norm.beta.assign(
            hf_wts[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()
        )

    keras_nlp_model.get_layer("pooled_dense").kernel.assign(
        hf_wts["pooler.dense.weight"].transpose(1, 0).numpy()
    )
    keras_nlp_model.get_layer("pooled_dense").bias.assign(
        hf_wts["pooler.dense.bias"].numpy()
    )

    # Save the model.
    print("\n-> Save KerasNLP model weights.")
    keras_nlp_model.save_weights(os.path.join(FLAGS.preset, "model.h5"))

    return keras_nlp_model


def extract_vocab(hf_tokenizer):
    spm_path = os.path.join(FLAGS.preset, "spiece.model")
    print(f"\n-> Save KerasNLP SPM vocabulary file to `{spm_path}`.")

    shutil.copyfile(
        transformers.utils.hub.get_file_from_repo(
            hf_tokenizer.name_or_path, "spiece.model"
        ),
        spm_path,
    )

    keras_nlp_tokenizer = keras_nlp.models.FNetTokenizer(
        proto=spm_path,
    )
    keras_nlp_preprocessor = keras_nlp.models.FNetPreprocessor(
        keras_nlp_tokenizer
    )

    print("-> Print MD5 checksum of the vocab files.")
    print(f"`{spm_path}` md5sum: ", get_md5_checksum(spm_path))

    return keras_nlp_preprocessor


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
    keras_nlp_output = keras_nlp_model.predict(keras_nlp_inputs)[
        "sequence_output"
    ]

    # HF
    hf_inputs = hf_tokenizer(
        sample_text, padding="max_length", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasNLP output:", keras_nlp_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print("Difference:", np.mean(keras_nlp_output - hf_output.detach().numpy()))

    # Show the MD5 checksum of the model weights.
    print(
        "Model md5sum: ",
        get_md5_checksum(os.path.join(FLAGS.preset, "model.h5")),
    )


def main(_):
    os.makedirs(FLAGS.preset)

    hf_model_name = PRESET_MAP[FLAGS.preset]

    print("\n-> Load HF model and HF tokenizer.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    keras_nlp_model = convert_checkpoints(hf_model)
    print("\n -> Load KerasNLP preprocessor.")
    keras_nlp_preprocessor = extract_vocab(hf_tokenizer)

    check_output(
        keras_nlp_preprocessor,
        keras_nlp_model,
        hf_tokenizer,
        hf_model,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
