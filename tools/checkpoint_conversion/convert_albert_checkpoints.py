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
    "albert_base_en_uncased": "albert-base-v2",
    "albert_large_en_uncased": "albert-large-v2",
    "albert_extra_large_en_uncased": "albert-xlarge-v2",
    "albert_extra_extra_large_en_uncased": "albert-xxlarge-v2",
}


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def convert_checkpoints(hf_model):
    print("\n-> Convert original weights to KerasNLP format.")

    print("\n-> Load KerasNLP model.")
    keras_nlp_model = keras_nlp.models.AlbertBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(list(hf_wts.keys()))

    num_heads = keras_nlp_model.num_heads
    hidden_dim = keras_nlp_model.hidden_dim

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
        hf_wts["encoder.embedding_hidden_mapping_in.weight"].T
    )
    keras_nlp_model.get_layer("embedding_projection").bias.assign(
        hf_wts["encoder.embedding_hidden_mapping_in.bias"]
    )

    for i in range(keras_nlp_model.num_groups):
        for j in range(keras_nlp_model.num_inner_repetitions):
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._query_dense.kernel.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.query.weight"
                ]
                .transpose(1, 0)
                .reshape((hidden_dim, num_heads, -1))
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._query_dense.bias.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.query.bias"
                ]
                .reshape((num_heads, -1))
                .numpy()
            )

            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._key_dense.kernel.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.key.weight"
                ]
                .transpose(1, 0)
                .reshape((hidden_dim, num_heads, -1))
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._key_dense.bias.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.key.bias"
                ]
                .reshape((num_heads, -1))
                .numpy()
            )

            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._value_dense.kernel.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.value.weight"
                ]
                .transpose(1, 0)
                .reshape((hidden_dim, num_heads, -1))
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._value_dense.bias.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.value.bias"
                ]
                .reshape((num_heads, -1))
                .numpy()
            )

            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._output_dense.kernel.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.dense.weight"
                ]
                .transpose(1, 0)
                .reshape((num_heads, -1, hidden_dim))
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layer._output_dense.bias.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.dense.bias"
                ].numpy()
            )

            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layernorm.gamma.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.LayerNorm.weight"
                ].numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._self_attention_layernorm.beta.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.attention.LayerNorm.bias"
                ].numpy()
            )

            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._feedforward_intermediate_dense.kernel.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.ffn.weight"
                ]
                .transpose(1, 0)
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._feedforward_intermediate_dense.bias.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.ffn.bias"
                ].numpy()
            )

            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._feedforward_output_dense.kernel.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.ffn_output.weight"
                ]
                .transpose(1, 0)
                .numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._feedforward_output_dense.bias.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.ffn_output.bias"
                ].numpy()
            )

            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._feedforward_layernorm.gamma.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.full_layer_layer_norm.weight"
                ].numpy()
            )
            keras_nlp_model.get_layer(
                f"group_{i}_inner_layer_{j}"
            )._feedforward_layernorm.beta.assign(
                hf_wts[
                    f"encoder.albert_layer_groups.{i}.albert_layers.{j}.full_layer_layer_norm.bias"
                ].numpy()
            )

    keras_nlp_model.get_layer("pooled_dense").kernel.assign(
        hf_wts["pooler.weight"].transpose(1, 0).numpy()
    )
    keras_nlp_model.get_layer("pooled_dense").bias.assign(
        hf_wts["pooler.bias"].numpy()
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

    keras_nlp_tokenizer = keras_nlp.models.AlbertTokenizer(
        proto=spm_path,
    )
    keras_nlp_preprocessor = keras_nlp.models.AlbertPreprocessor(
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
