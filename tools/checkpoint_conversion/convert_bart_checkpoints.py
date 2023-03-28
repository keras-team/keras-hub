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
    "bart_base_en": "facebook/bart-base",
    "bart_large_en": "facebook/bart-large",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def convert_checkpoints(hf_model):
    print("\n-> Convert original weights to KerasNLP format.")

    print("\n-> Load KerasNLP model.")
    keras_nlp_model = keras_nlp.models.BartBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(list(hf_wts.keys()))

    hidden_dim = keras_nlp_model.hidden_dim
    num_heads = keras_nlp_model.num_heads

    # Token embedding weights shared by encoder and decoder.
    keras_nlp_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["shared.weight"]
    )

    # Encoder weights.
    keras_nlp_model.get_layer(
        "encoder_position_embedding"
    ).position_embeddings.assign(hf_wts["encoder.embed_positions.weight"][2:])

    keras_nlp_model.get_layer("encoder_embeddings_layer_norm").gamma.assign(
        hf_wts["encoder.layernorm_embedding.weight"]
    )
    keras_nlp_model.get_layer("encoder_embeddings_layer_norm").beta.assign(
        hf_wts["encoder.layernorm_embedding.bias"]
    )

    for i in range(keras_nlp_model.num_layers):
        # Self-attention.
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.q_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.q_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.k_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.k_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.v_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.v_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.out_proj.weight"]
            .transpose(1, 0)
            .reshape((num_heads, -1, hidden_dim))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.out_proj.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            hf_wts[f"encoder.layers.{i}.self_attn_layer_norm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            hf_wts[f"encoder.layers.{i}.self_attn_layer_norm.bias"].numpy()
        )

        # Post self-attention layers.
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.fc1.weight"].transpose(1, 0).numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.fc1.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.fc2.weight"].transpose(1, 0).numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.fc2.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            hf_wts[f"encoder.layers.{i}.final_layer_norm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            hf_wts[f"encoder.layers.{i}.final_layer_norm.bias"].numpy()
        )

    # Decoder weights.

    keras_nlp_model.get_layer(
        "decoder_position_embedding"
    ).position_embeddings.assign(hf_wts["decoder.embed_positions.weight"][2:])

    keras_nlp_model.get_layer("decoder_embeddings_layer_norm").gamma.assign(
        hf_wts["decoder.layernorm_embedding.weight"]
    )
    keras_nlp_model.get_layer("decoder_embeddings_layer_norm").beta.assign(
        hf_wts["decoder.layernorm_embedding.bias"]
    )

    for i in range(keras_nlp_model.num_layers):
        # Self-attention.
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.q_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.q_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.k_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.k_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.v_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.v_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.out_proj.weight"]
            .transpose(1, 0)
            .reshape((num_heads, -1, hidden_dim))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.out_proj.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            hf_wts[f"decoder.layers.{i}.self_attn_layer_norm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            hf_wts[f"decoder.layers.{i}.self_attn_layer_norm.bias"].numpy()
        )

        # Cross-attention.
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._query_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.q_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._query_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.q_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._key_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.k_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._key_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.k_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._value_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.v_proj.weight"]
            .transpose(1, 0)
            .reshape((hidden_dim, num_heads, -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._value_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.v_proj.bias"]
            .reshape((num_heads, -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._output_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.out_proj.weight"]
            .transpose(1, 0)
            .reshape((num_heads, -1, hidden_dim))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._output_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.out_proj.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layernorm.gamma.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn_layer_norm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layernorm.beta.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn_layer_norm.bias"].numpy()
        )

        # Post self-attention and cross-attention layers.
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.fc1.weight"].transpose(1, 0).numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.fc1.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.fc2.weight"].transpose(1, 0).numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.fc2.bias"].numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            hf_wts[f"decoder.layers.{i}.final_layer_norm.weight"].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            hf_wts[f"decoder.layers.{i}.final_layer_norm.bias"].numpy()
        )

    # Save the model.
    print("\n-> Save KerasNLP model weights.")
    keras_nlp_model.save_weights(os.path.join(FLAGS.preset, "model.h5"))

    return keras_nlp_model


def extract_vocab(hf_tokenizer):
    vocabulary_path = os.path.join(FLAGS.preset, "vocab.json")
    merges_path = os.path.join(FLAGS.preset, "merges.txt")
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

    keras_nlp_tokenizer = keras_nlp.models.BartTokenizer(
        vocabulary=vocabulary_path, merges=merges_path
    )

    print("-> Print MD5 checksum of the vocab files.")
    print(f"`{vocabulary_path}` md5sum: ", get_md5_checksum(vocabulary_path))
    print(f"`{merges_path}` md5sum: ", get_md5_checksum(merges_path))

    return keras_nlp_tokenizer


def check_output(
    keras_nlp_tokenizer,
    keras_nlp_model,
    hf_tokenizer,
    hf_model,
):
    print("\n-> Check the outputs.")
    enc_sample_text = [
        "cricket is awesome, easily the best sport in the world!"
    ]
    dec_sample_text = [
        "football is good too, but nowhere near as good as cricket."
    ]

    # KerasNLP
    keras_nlp_enc_token_ids = keras_nlp_tokenizer(
        tf.constant(enc_sample_text)
    ).to_tensor()
    keras_nlp_enc_token_ids = tf.concat(
        [
            tf.constant([[keras_nlp_tokenizer.start_token_id]]),
            keras_nlp_enc_token_ids,
            tf.constant([[keras_nlp_tokenizer.end_token_id]]),
        ],
        axis=-1,
    )
    keras_nlp_dec_token_ids = keras_nlp_tokenizer(
        tf.constant(dec_sample_text)
    ).to_tensor()
    keras_nlp_dec_token_ids = tf.concat(
        [
            tf.constant([[keras_nlp_tokenizer.start_token_id]]),
            keras_nlp_dec_token_ids,
            tf.constant([[keras_nlp_tokenizer.end_token_id]]),
        ],
        axis=-1,
    )
    keras_nlp_inputs = {
        "encoder_token_ids": keras_nlp_enc_token_ids,
        "encoder_padding_mask": keras_nlp_enc_token_ids
        != keras_nlp_tokenizer.pad_token_id,
        "decoder_token_ids": keras_nlp_dec_token_ids,
        "decoder_padding_mask": keras_nlp_dec_token_ids
        != keras_nlp_tokenizer.pad_token_id,
    }
    keras_nlp_output = keras_nlp_model.predict(keras_nlp_inputs)

    # HF
    hf_enc_inputs = hf_tokenizer(enc_sample_text, return_tensors="pt")
    hf_dec_inputs = hf_tokenizer(dec_sample_text, return_tensors="pt")

    hf_output = hf_model(
        **hf_enc_inputs,
        decoder_input_ids=hf_dec_inputs["input_ids"],
        decoder_attention_mask=hf_dec_inputs["attention_mask"],
    )

    print("Encoder Outputs:")
    print(
        "KerasNLP output:",
        keras_nlp_output["encoder_sequence_output"][0, 0, :10],
    )
    print("HF output:", hf_output.encoder_last_hidden_state[0, 0, :10])
    print(
        "Difference:",
        np.mean(
            keras_nlp_output["encoder_sequence_output"]
            - hf_output.encoder_last_hidden_state.detach().numpy()
        ),
    )

    print("Decoder Outputs:")
    print(
        "KerasNLP output:",
        keras_nlp_output["decoder_sequence_output"][0, 0, :10],
    )
    print("HF output:", hf_output.last_hidden_state[0, 0, :10])
    print(
        "Difference:",
        np.mean(
            keras_nlp_output["decoder_sequence_output"]
            - hf_output.last_hidden_state.detach().numpy()
        ),
    )

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
    print("\n -> Load KerasNLP tokenizer.")
    keras_nlp_tokenizer = extract_vocab(hf_tokenizer)

    check_output(
        keras_nlp_tokenizer,
        keras_nlp_model,
        hf_tokenizer,
        hf_model,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
