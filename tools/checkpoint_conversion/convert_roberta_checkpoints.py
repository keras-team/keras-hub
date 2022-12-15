# Copyright 2022 The KerasNLP Authors
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
import torch
import transformers
from absl import app
from absl import flags
from tensorflow import keras

import keras_nlp
from tools.checkpoint_conversion.checkpoint_conversion_utils import (
    get_md5_checksum,
)

PRESET_MAP = {
    "roberta_base": ("roberta.base", "roberta-base"),
    "roberta_large": ("roberta.large", "roberta-large"),
}

DOWNLOAD_SCRIPT_URL = "https://dl.fbaipublicfiles.com/fairseq/models/{}.tar.gz"

EXTRACT_DIR = "./{}"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)


def download_model(preset, size, hf_model_name):
    print("-> Download original weights.")
    extract_dir = EXTRACT_DIR.format(size)
    archive_file_path = keras.utils.get_file(
        fname=None,
        origin=DOWNLOAD_SCRIPT_URL.format(size),
        cache_subdir=os.path.join("checkpoint_conversion", preset),
    )

    os.system(f"tar -xvf {archive_file_path}")

    # The original `tar.gz` file does not have the vocab files. Let's fetch
    # them from HF.
    vocabulary_path = keras.utils.get_file(
        fname=None,
        origin=f"https://huggingface.co/{hf_model_name}/raw/main/vocab.json",
    )
    shutil.copy(vocabulary_path, extract_dir)
    merges_path = keras.utils.get_file(
        fname=None,
        origin=f"https://huggingface.co/{hf_model_name}/raw/main/merges.txt",
    )
    shutil.copy(merges_path, extract_dir)


def convert_checkpoints(preset, size):
    print("\n-> Convert original weights to KerasNLP format.")
    # RoBERTa paths.
    extract_dir = EXTRACT_DIR.format(size)
    checkpoint_path = os.path.join(extract_dir, "model.pt")

    # Load PyTorch RoBERTa checkpoint.
    pt_ckpt = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    pt_cfg = pt_ckpt["args"]
    pt_model = pt_ckpt["model"]

    cfg = {
        "num_layers": pt_cfg.encoder_layers,
        "num_heads": pt_cfg.encoder_attention_heads,
        "hidden_dim": pt_cfg.encoder_embed_dim,
        "intermediate_dim": pt_cfg.encoder_ffn_embed_dim,
        "dropout": pt_cfg.dropout,
        "max_sequence_length": pt_cfg.max_positions,
        "vocab_size": (
            pt_model["decoder.sentence_encoder.embed_tokens.weight"]
            .numpy()
            .shape[0]
        ),
    }
    print("Config:", cfg)

    keras_nlp_model = keras_nlp.models.RobertaBackbone.from_preset(
        preset, load_weights=False
    )

    # Embedding Layer.
    keras_nlp_model.get_layer("embeddings").token_embedding.embeddings.assign(
        pt_model["decoder.sentence_encoder.embed_tokens.weight"].numpy()
    )
    keras_nlp_model.get_layer(
        "embeddings"
    ).position_embedding.position_embeddings.assign(
        pt_model["decoder.sentence_encoder.embed_positions.weight"].numpy()[
            2:, :
        ]
    )

    # Embedding LayerNorm.
    keras_nlp_model.get_layer("embeddings_layer_norm").gamma.assign(
        pt_model["decoder.sentence_encoder.emb_layer_norm.weight"].numpy()
    )
    keras_nlp_model.get_layer("embeddings_layer_norm").beta.assign(
        pt_model["decoder.sentence_encoder.emb_layer_norm.bias"].numpy()
    )

    # The QKV weights in the original checkpoint are present as one single
    # dense layer of shape `(3 * cfg["hidden_dim"], cfg["hidden_dim"])`. Our
    # model has three separate dense layers for each of QKV. Hence, we need to
    # split the original QKV weights into three chunks.
    range_1 = (0, cfg["hidden_dim"])
    range_2 = (cfg["hidden_dim"], 2 * cfg["hidden_dim"])
    range_3 = (2 * cfg["hidden_dim"], 3 * cfg["hidden_dim"])
    # Transformer layers.
    for i in range(keras_nlp_model.num_layers):
        q_k_v_wts = (
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.self_attn.in_proj_weight"
            ]
            .numpy()
            .T
        )
        q_k_v_bias = (
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.self_attn.in_proj_bias"
            ]
            .numpy()
            .T
        )

        # Query
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.kernel.assign(
            q_k_v_wts[:, range_1[0] : range_1[1]].reshape(
                (cfg["hidden_dim"], cfg["num_heads"], -1)
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._query_dense.bias.assign(
            q_k_v_bias[range_1[0] : range_1[1]].reshape((cfg["num_heads"], -1))
        )

        # Key
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.kernel.assign(
            q_k_v_wts[:, range_2[0] : range_2[1]].reshape(
                (cfg["hidden_dim"], cfg["num_heads"], -1)
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._key_dense.bias.assign(
            q_k_v_bias[range_2[0] : range_2[1]].reshape((cfg["num_heads"], -1))
        )

        # Value
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.kernel.assign(
            q_k_v_wts[:, range_3[0] : range_3[1]].reshape(
                (cfg["hidden_dim"], cfg["num_heads"], -1)
            )
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._value_dense.bias.assign(
            q_k_v_bias[range_3[0] : range_3[1]].reshape((cfg["num_heads"], -1))
        )

        # Attention output
        attn_output_wts = (
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.self_attn.out_proj.weight"
            ]
            .numpy()
            .T
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.kernel.assign(
            attn_output_wts.reshape((cfg["num_heads"], -1, cfg["hidden_dim"]))
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layer._output_dense.bias.assign(
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.self_attn.out_proj.bias"
            ].numpy()
        )

        # Attention LayerNorm
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.gamma.assign(
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.self_attn_layer_norm.weight"
            ].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._self_attention_layernorm.beta.assign(
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.self_attn_layer_norm.bias"
            ].numpy()
        )

        # Intermediate FF layer
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.kernel.assign(
            pt_model[f"decoder.sentence_encoder.layers.{i}.fc1.weight"]
            .numpy()
            .T
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_intermediate_dense.bias.assign(
            pt_model[f"decoder.sentence_encoder.layers.{i}.fc1.bias"].numpy()
        )

        # Output dense layer
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.kernel.assign(
            pt_model[f"decoder.sentence_encoder.layers.{i}.fc2.weight"]
            .numpy()
            .T
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_output_dense.bias.assign(
            pt_model[f"decoder.sentence_encoder.layers.{i}.fc2.bias"].numpy()
        )

        # FF LayerNorm
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.gamma.assign(
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.final_layer_norm.weight"
            ].numpy()
        )
        keras_nlp_model.get_layer(
            f"transformer_layer_{i}"
        )._feedforward_layernorm.beta.assign(
            pt_model[
                f"decoder.sentence_encoder.layers.{i}.final_layer_norm.bias"
            ].numpy()
        )

    # Save the model.
    print(f"\n-> Save KerasNLP model weights to `{preset}.h5`.")
    keras_nlp_model.save_weights(f"{preset}.h5")

    return keras_nlp_model


def define_preprocessor(preset, hf_model_name, size):
    print("\n-> Define the tokenizers.")
    extract_dir = EXTRACT_DIR.format(size)
    vocabulary_path = os.path.join(extract_dir, "vocab.json")
    merges_path = os.path.join(extract_dir, "merges.txt")

    keras_nlp_tokenizer = keras_nlp.models.RobertaTokenizer(
        vocabulary=vocabulary_path, merges=merges_path
    )
    keras_nlp_preprocessor = keras_nlp.models.RobertaPreprocessor(
        keras_nlp_tokenizer
    )

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    print("\n-> Print MD5 checksum of the vocab files.")
    print(f"`{vocabulary_path}` md5sum: ", get_md5_checksum(vocabulary_path))
    print(f"`{merges_path}` md5sum: ", get_md5_checksum(merges_path))

    return keras_nlp_preprocessor, hf_tokenizer


def check_output(
    preset,
    keras_nlp_model,
    keras_nlp_preprocessor,
    hf_model,
    hf_tokenizer,
):
    print("\n-> Check the outputs.")
    input_str = ["the quick brown fox ran, galloped and jumped."]

    # KerasNLP
    keras_nlp_inputs = keras_nlp_preprocessor(tf.constant(input_str))
    keras_nlp_output = keras_nlp_model.predict(keras_nlp_inputs)

    # HF
    hf_inputs = hf_tokenizer(
        input_str, padding="max_length", return_tensors="pt"
    )
    hf_output = hf_model(**hf_inputs).last_hidden_state

    print("KerasNLP output:", keras_nlp_output[0, 0, :10])
    print("HF output:", hf_output[0, 0, :10])
    print("Difference:", np.mean(keras_nlp_output - hf_output.detach().numpy()))

    # Show the MD5 checksum of the model weights.
    print("Model md5sum: ", get_md5_checksum(f"./{preset}.h5"))

    return keras_nlp_output


def main(_):
    assert (
        FLAGS.preset in PRESET_MAP.keys()
    ), f'Invalid preset {FLAGS.preset}. Must be one of {",".join(PRESET_MAP.keys())}'
    size = PRESET_MAP[FLAGS.preset][0]
    hf_model_name = PRESET_MAP[FLAGS.preset][1]

    download_model(FLAGS.preset, size, hf_model_name)

    keras_nlp_model = convert_checkpoints(FLAGS.preset, size)

    print("\n-> Load HF model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)
    hf_model.eval()

    keras_nlp_preprocessor, hf_tokenizer = define_preprocessor(
        FLAGS.preset, hf_model_name, size
    )

    check_output(
        FLAGS.preset,
        keras_nlp_model,
        keras_nlp_preprocessor,
        hf_model,
        hf_tokenizer,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
