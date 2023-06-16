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
    "pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def convert_checkpoints(hf_model):
    print("\n-> Convert original weights to KerasNLP format.")
    print("\n-> Load KerasNLP model.")

    keras_model = keras_nlp.models.GPTNeoXBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )
    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(list(hf_wts.keys()))

    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_model.embed_in.weight.detach().numpy()
    )

    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_model.embed_in.weight.detach().numpy()
    )

    for ilayer in range(keras_model.num_layers):
        # attention layer
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer._qkv_dense.kernel.assign(
            hf_wts[f"layers.{ilayer}.attention.query_key_value.weight"]
            .numpy()
            .T.reshape((keras_model.hidden_dim, keras_model.num_heads, -1))
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer._qkv_dense.bias.assign(
            hf_wts[f"layers.{ilayer}.attention.query_key_value.bias"].reshape(
                (keras_model.num_heads, -1)
            )
        )

        # Attention Dense
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"layers.{ilayer}.attention.dense.weight"].numpy().T
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_wts[f"layers.{ilayer}.attention.dense.bias"]
        )

        # LAYERNORM
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._input_layernorm.gamma.assign(
            hf_wts[f"layers.{ilayer}.input_layernorm.weight"]
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._input_layernorm.beta.assign(
            hf_wts[f"layers.{ilayer}.input_layernorm.bias"]
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layernorm.gamma.assign(
            hf_wts[f"layers.{ilayer}.post_attention_layernorm.weight"]
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layernorm.beta.assign(
            hf_wts[f"layers.{ilayer}.post_attention_layernorm.bias"]
        )

        # MLP
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_wts[f"layers.{ilayer}.mlp.dense_h_to_4h.weight"].numpy().T
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_wts[f"layers.{ilayer}.mlp.dense_h_to_4h.bias"]
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._feedforward_output_dense.kernel.assign(
            hf_wts[f"layers.{ilayer}.mlp.dense_4h_to_h.weight"].numpy().T
        )

        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._feedforward_output_dense.bias.assign(
            hf_wts[f"layers.{ilayer}.mlp.dense_4h_to_h.bias"]
        )

        # Rotary Embedding
        keras_model.get_layer(
            f"transformer_layer_{ilayer}"
        )._self_attention_layer.rotary_embedding.inverse_freq.assign(
            hf_wts[f"layers.{ilayer}.attention.rotary_emb.inv_freq"]
        )

    keras_model.get_layer("layer_norm").gamma.assign(
        hf_wts["final_layer_norm.weight"]
    )

    keras_model.get_layer("layer_norm").beta.assign(
        hf_wts["final_layer_norm.bias"]
    )

    # Save the model.
    print("\n-> Save KerasNLP model weights.")
    keras_model.save_weights(os.path.join(FLAGS.preset, "model.h5"))

    return keras_model


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
