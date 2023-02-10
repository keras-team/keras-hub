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

import keras_nlp
from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    get_md5_checksum,
)

PRESET_MAP = {
    "bart_base_en": "facebook/bart-base",
    "bart_large_en": "facebook/bart-large",
}

EXTRACT_DIR = "./{}"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def download_files(preset, hf_model_name):
    print("-> Download original vocabulary and config.")

    extract_dir = EXTRACT_DIR.format(preset)
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
    vocab_path = os.path.join(extract_dir, "vocab.json")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/raw/main/vocab.json"
    )
    open(vocab_path, "wb").write(response.content)
    print(f"`{vocab_path}`")

    # Merges file.
    merges_path = os.path.join(extract_dir, "merges.txt")
    response = requests.get(
        f"https://huggingface.co/{hf_model_name}/raw/main/merges.txt"
    )
    open(merges_path, "wb").write(response.content)
    print(f"`{merges_path}`")


def define_tokenizer(preset, hf_model_name):
    print("\n-> Define the tokenizers.")
    extract_dir = EXTRACT_DIR.format(preset)
    vocab_path = os.path.join(extract_dir, "vocab.json")
    merges_path = os.path.join(extract_dir, "merges.txt")

    keras_nlp_tokenizer = keras_nlp.models.BartTokenizer(
        vocabulary=vocab_path, merges=merges_path
    )

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    print("\n-> Print MD5 checksum of the vocab files.")
    print(f"`{vocab_path}` md5sum: ", get_md5_checksum(vocab_path))
    print(f"`{merges_path}` md5sum: ", get_md5_checksum(merges_path))

    return keras_nlp_tokenizer, hf_tokenizer


def convert_checkpoints(preset, keras_nlp_model, hf_model):
    print("\n-> Convert original weights to KerasNLP format.")

    extract_dir = EXTRACT_DIR.format(preset)
    config_path = os.path.join(extract_dir, "config.json")

    # Build config.
    cfg = {}
    with open(config_path, "r") as hf_cfg_handler:
        hf_cfg = json.load(hf_cfg_handler)
    cfg["vocabulary_size"] = hf_cfg["vocab_size"]
    cfg["num_layers"] = hf_cfg["num_hidden_layers"]
    cfg["num_heads"] = hf_cfg["encoder_attention_heads"]
    cfg["hidden_dim"] = hf_cfg["d_model"]
    cfg["intermediate_dim"] = hf_cfg["encoder_ffn_dim"]
    cfg["dropout"] = hf_cfg["dropout"]
    cfg["max_sequence_length"] = hf_cfg["max_position_embeddings"]
    print("Config:", cfg)

    hf_wts = hf_model.state_dict()
    print("Original weights:")
    print(list(hf_wts.keys()))

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
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.q_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.k_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.k_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.v_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.v_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_encoder_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"encoder.layers.{i}.self_attn.out_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
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
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.q_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.k_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.k_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.v_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.v_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.self_attn.out_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
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
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._query_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.q_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._key_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.k_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._key_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.k_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._value_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.v_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["hidden_dim"], cfg["num_heads"], -1))
            .numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._value_dense.bias.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.v_proj.bias"]
            .reshape((cfg["num_heads"], -1))
            .numpy()
        )

        keras_nlp_model.get_layer(
            f"transformer_decoder_layer_{i}"
        )._cross_attention_layer._output_dense.kernel.assign(
            hf_wts[f"decoder.layers.{i}.encoder_attn.out_proj.weight"]
            .transpose(1, 0)
            .reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
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
    print(f"\n-> Save KerasNLP model weights to `{preset}.h5`.")
    keras_nlp_model.save_weights(f"{preset}.h5")

    return keras_nlp_model


def check_output(
    preset,
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
    print("Model md5sum: ", get_md5_checksum(f"./{preset}.h5"))


def main(_):
    hf_model_name = PRESET_MAP[FLAGS.preset]

    download_files(FLAGS.preset, hf_model_name)

    keras_nlp_tokenizer, hf_tokenizer = define_tokenizer(
        FLAGS.preset, hf_model_name
    )

    print("\n-> Load KerasNLP model.")
    keras_nlp_model = keras_nlp.models.BartBackbone.from_preset(
        FLAGS.preset, load_weights=False
    )

    print("\n-> Load HF model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    keras_nlp_model = convert_checkpoints(
        FLAGS.preset, keras_nlp_model, hf_model
    )

    check_output(
        FLAGS.preset,
        keras_nlp_tokenizer,
        keras_nlp_model,
        hf_tokenizer,
        hf_model,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
