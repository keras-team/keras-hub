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
    "opt_125m_en": "facebook/opt-125m",
    "opt_1.3b_en": "facebook/opt-1.3b",
    "opt_2.7b_en": "facebook/opt-2.7b",
    "opt_6.7b_en": "facebook/opt-6.7b",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def convert_weights(hf_model):
    print("\n-> Convert original weights to KerasNLP format.")

    # Load PyTorch OPT checkpoint.
    keras_nlp_model = keras_nlp.models.OPTBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    # Token embedding.
    keras_nlp_model.get_layer("embeddings").token_embedding.embeddings.assign(
        hf_model.model.decoder.embed_tokens.weight
    )
    # Position embedding.
    keras_nlp_model.get_layer(
        "embeddings"
    ).position_embedding.position_embeddings.assign(
        hf_model.model.decoder.embed_positions.weight[2:, :]
    )

    num_heads = keras_nlp_model.num_heads
    hidden_dim = keras_nlp_model.hidden_dim

    # Transformer layers.
    for i in range(keras_nlp_model.num_layers):
        # Self-attention.
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            tf.reshape(
                hf_model.model.decoder.layers[i].self_attn.q_proj.kernel,
                (hidden_dim, num_heads, -1),
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            tf.reshape(
                hf_model.model.decoder.layers[i].self_attn.q_proj.bias,
                (num_heads, -1),
            )
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            tf.reshape(
                hf_model.model.decoder.layers[i].self_attn.k_proj.kernel,
                (hidden_dim, num_heads, -1),
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            tf.reshape(
                hf_model.model.decoder.layers[i].self_attn.k_proj.bias,
                (num_heads, -1),
            )
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            tf.reshape(
                hf_model.model.decoder.layers[i].self_attn.v_proj.kernel,
                (hidden_dim, num_heads, -1),
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            tf.reshape(
                hf_model.model.decoder.layers[i].self_attn.v_proj.bias,
                (num_heads, -1),
            )
        )

        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            tf.reshape(
                hf_model.model.decoder.layers[i].self_attn.out_proj.kernel,
                (num_heads, -1, hidden_dim),
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_model.model.decoder.layers[i].self_attn.out_proj.bias,
        )

        # Attention LayerNorm
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            hf_model.model.decoder.layers[i].self_attn_layer_norm.gamma
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            hf_model.model.decoder.layers[i].self_attn_layer_norm.beta
        )

        # Intermediate FF layer
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_model.model.decoder.layers[i].fc1.kernel
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_model.model.decoder.layers[i].fc1.bias
        )

        # Output dense layer
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            hf_model.model.decoder.layers[i].fc2.kernel
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            hf_model.model.decoder.layers[i].fc2.bias
        )

        # FF LayerNorm
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            hf_model.model.decoder.layers[i].final_layer_norm.gamma
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            hf_model.model.decoder.layers[i].final_layer_norm.beta
        )

    # Output LayerNorm
    keras_nlp_model.get_layer("layer_norm").gamma.assign(
        hf_model.model.decoder.final_layer_norm.gamma
    )
    keras_nlp_model.get_layer("layer_norm").beta.assign(
        hf_model.model.decoder.final_layer_norm.beta
    )

    # Save the model.
    model_path = f"./{FLAGS.preset}/model.h5"
    print(f"-> Save KerasNLP model weights to `{model_path}`.")
    keras_nlp_model.save_weights(model_path)
    print("-> Print MD5 checksum of the model weights files.")
    print(f"`{model_path}` md5sum: ", get_md5_checksum(model_path))

    return keras_nlp_model


def extract_vocab(hf_tokenizer):
    vocabulary_path = f"./{FLAGS.preset}/vocab.json"
    merges_path = f"./{FLAGS.preset}/merges.txt"
    print(f"\n-> Save KerasNLP vocab to `{vocabulary_path}`.")
    print(f"-> Save KerasNLP merges to `{merges_path}`.")

    # Huggingface has a save_vocabulary function but it's not byte-for-byte
    # with the source. Instead copy the original downloaded file directly.
    shutil.copyfile(
        transformers.utils.hub.get_file_from_repo(
            hf_tokenizer.name_or_path, "vocab.json"
        ),
        vocabulary_path,
    )
    shutil.copyfile(
        transformers.utils.hub.get_file_from_repo(
            hf_tokenizer.name_or_path, "merges.txt"
        ),
        merges_path,
    )

    keras_nlp_tokenizer = keras_nlp.models.OPTTokenizer(
        vocabulary=vocabulary_path, merges=merges_path
    )

    print("-> Print MD5 checksum of the vocab files.")
    print(f"`{vocabulary_path}` md5sum: ", get_md5_checksum(vocabulary_path))
    print(f"`{merges_path}` md5sum: ", get_md5_checksum(merges_path))

    return keras_nlp_tokenizer


def check_output(
    keras_nlp_model,
    keras_nlp_tokenizer,
    hf_model,
    hf_tokenizer,
):
    print("\n-> Check the outputs.")
    input_str = ["the quick brown fox ran, galloped and jumped."]

    sequence_length = 16
    packer = keras_nlp.layers.StartEndPacker(
        sequence_length=sequence_length,
        start_value=keras_nlp_tokenizer.start_token_id,
        pad_value=keras_nlp_tokenizer.pad_token_id,
    )

    # KerasNLP
    token_ids = packer(keras_nlp_tokenizer(input_str))
    padding_mask = token_ids != keras_nlp_tokenizer.pad_token_id
    keras_nlp_inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }
    keras_nlp_output = keras_nlp_model(keras_nlp_inputs)

    # HF
    hf_inputs = hf_tokenizer(
        input_str,
        padding="max_length",
        max_length=sequence_length,
        return_tensors="tf",
    )
    hf_output = hf_model(
        **hf_inputs, return_dict=True, output_hidden_states=True
    )

    # Compare tokenized inputs. This should be a compete match.
    print("KerasNLP inputs:", keras_nlp_inputs)
    print("HF inputs:", hf_inputs)

    # Compare outputs, this should match closely, though not exactly.
    hf_output = hf_output.last_hidden_state
    print("KerasNLP output:", keras_nlp_output[0, 0, :5])
    print("HF output:", hf_output[0, 0, :5])
    difference = keras_nlp_output - hf_output
    difference_non_padding = tf.gather_nd(difference, tf.where(padding_mask))
    print("Difference:", np.mean(difference_non_padding))


def main(_):
    hf_id = PRESET_MAP[FLAGS.preset]
    os.mkdir(f"./{FLAGS.preset}")

    print("\n-> Load HF model.")
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_id)
    hf_model = transformers.TFAutoModel.from_pretrained(hf_id)

    keras_nlp_tokenizer = extract_vocab(hf_tokenizer)
    keras_nlp_model = convert_weights(hf_model)

    check_output(
        keras_nlp_model,
        keras_nlp_tokenizer,
        hf_model,
        hf_tokenizer,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
