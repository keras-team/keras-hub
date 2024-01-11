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

import numpy as np
import torch
import transformers
from absl import app
from absl import flags
from checkpoint_conversion_utils import get_md5_checksum

from keras_nlp.models.bloom.bloom_backbone import BloomBackbone

FLAGS = flags.FLAGS

PRESET_MAP = {
    "bloom_tiny": "bigscience/bloom-560m",
    "bloom_extra_small": "bigscience/bloom-1b1",
    "bloom_small": "bigscience/bloom-1b7",
    "bloom_meduim": "bigscience/bloom-3b",
    "bloom_large": "bigscience/bloom-7b1",
    "bloom_extra_large": "bigscience/bloom",
}

flags.DEFINE_string(
    "preset", None, f'Must be one of {",".join(PRESET_MAP.keys())}'
)
flags.mark_flag_as_required("preset")


def convert_checkpoints(hf_model):
    # get huggingface model configuration.
    hf_config = hf_model.config.to_dict()

    cfg = {}
    cfg["vocabulary_size"] = hf_config["vocab_size"]
    cfg["num_layers"] = hf_config["n_layer"]
    cfg["num_heads"] = hf_config["n_head"]
    cfg["hidden_dim"] = hf_config["hidden_size"]
    cfg["intermediate_dim"] = hf_config["hidden_size"] * 4
    cfg["dropout"] = hf_config["hidden_dropout"]
    cfg["layer_norm_epsilon"] = hf_config["layer_norm_epsilon"]

    hidden_dim = cfg["hidden_dim"]
    num_heads = cfg["num_heads"]
    head_dim = hidden_dim // num_heads

    # Intialize Bloom model with the weights.
    keras_model = BloomBackbone(**cfg)

    # get huggingface model weights.
    hf_wts = hf_model.state_dict()

    # assign huggingface weights to the keras model.
    # Embedding layer.
    keras_model.get_layer("token_embedding").embeddings.assign(
        hf_wts["word_embeddings.weight"]
    )
    # LayerNorm.
    keras_model.get_layer("token_embedding_layernorm").gamma.assign(
        hf_wts["word_embeddings_layernorm.weight"]
    )
    keras_model.get_layer("token_embedding_layernorm").beta.assign(
        hf_wts["word_embeddings_layernorm.bias"]
    )

    keras_model.get_layer("final_layernorm").gamma.assign(hf_wts["ln_f.weight"])
    keras_model.get_layer("final_layernorm").beta.assign(hf_wts["ln_f.bias"])

    # Decoder layers.
    for i in range(cfg["num_layers"]):
        decoder_layer = keras_model.get_layer(f"transformer_layer_{i}")
        # LayrNorm.
        decoder_layer._pre_attention_layernorm.gamma.assign(
            hf_wts[f"h.{i}.input_layernorm.weight"]
        )
        decoder_layer._pre_attention_layernorm.beta.assign(
            hf_wts[f"h.{i}.input_layernorm.bias"]
        )
        decoder_layer._post_attention_layernorm.gamma.assign(
            hf_wts[f"h.{i}.post_attention_layernorm.weight"]
        )
        decoder_layer._post_attention_layernorm.beta.assign(
            hf_wts[f"h.{i}.post_attention_layernorm.bias"]
        )

        # Attention layer.
        attention_layer = decoder_layer._self_attention_layer

        fused_qkv_kernal = hf_wts[
            f"h.{i}.self_attention.query_key_value.weight"
        ].T
        fused_qkv_kernal = fused_qkv_kernal.view(
            hidden_dim, num_heads, 3, head_dim
        )
        query_kernal = fused_qkv_kernal[..., 0, :]
        key_kernal = fused_qkv_kernal[..., 1, :]
        value_kernl = fused_qkv_kernal[..., 2, :]

        fused_qkv_bais = hf_wts[f"h.{i}.self_attention.query_key_value.bias"]
        fused_qkv_bais = fused_qkv_bais.view(num_heads, 3, head_dim)
        query_bais = fused_qkv_bais[:, 0, :]
        key_bais = fused_qkv_bais[:, 1, :]
        value_bais = fused_qkv_bais[:, 2, :]

        attention_layer._query_dense.kernel.assign(query_kernal)
        attention_layer._query_dense.bias.assign(query_bais)
        attention_layer._key_dense.kernel.assign(key_kernal)
        attention_layer._key_dense.bias.assign(key_bais)
        attention_layer._value_dense.kernel.assign(value_kernl)
        attention_layer._value_dense.bias.assign(value_bais)

        attention_layer._output_dense.kernel.assign(
            hf_wts[f"h.{i}.self_attention.dense.weight"].T
        )
        attention_layer._output_dense.bias.assign(
            hf_wts[f"h.{i}.self_attention.dense.bias"]
        )

        # mlp.
        decoder_layer._mlp_intermediate_dense.kernel.assign(
            hf_wts[f"h.{i}.mlp.dense_h_to_4h.weight"].T
        )
        decoder_layer._mlp_intermediate_dense.bias.assign(
            hf_wts[f"h.{i}.mlp.dense_h_to_4h.bias"]
        )
        decoder_layer._mlp_output_dense.kernel.assign(
            hf_wts[f"h.{i}.mlp.dense_4h_to_h.weight"].T
        )
        decoder_layer._mlp_output_dense.bias.assign(
            hf_wts[f"h.{i}.mlp.dense_4h_to_h.bias"]
        )

    # Save the model.
    print(f"\n-> Saving KerasNLP model weights to `{FLAGS.preset}.weights.h5`.")
    keras_model.save_weights(f"{FLAGS.preset}.weights.h5")

    return keras_model


def check_output(keras_model, hf_model):
    hf_model_input = {
        "input_ids": torch.tensor([[59414, 15, 2670, 35433, 632, 207595]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
    }

    hf_model_outputs = hf_model(**hf_model_input)
    hf_model_outputs = hf_model_outputs.last_hidden_state
    hf_model_outputs = hf_model_outputs.detach().numpy()

    keras_model_input = {
        "token_ids": torch.tensor([[59414, 15, 2670, 35433, 632, 207595]]),
        "padding_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
    }

    keras_model_outputs = keras_model.predict(keras_model_input)

    # Comparing the outputs.
    print("KerasNLP output:", keras_model_outputs[0, 0, :10])
    print("HF output:", hf_model_outputs[0, 0, :10])
    print("Difference:", np.mean(keras_model_outputs - hf_model_outputs))

    # Show the MD5 checksum of the model weights.
    print("Model md5sum: ", get_md5_checksum(f"./{FLAGS.preset}.weights.h5"))


def main(_):
    assert (
        FLAGS.preset in PRESET_MAP.keys()
    ), f'Invalid preset {FLAGS.preset}. Must be one of {",".join(PRESET_MAP.keys())}'

    hf_model_name = PRESET_MAP[FLAGS.preset]

    print("\n-> Loading HF model.")
    hf_model = transformers.AutoModel.from_pretrained(hf_model_name)

    print("\n-> Converting model checkpoint.")
    keras_model = convert_checkpoints(hf_model)

    print("\n-> Checking keras model output.")
    check_output(keras_model, hf_model)


if __name__ == "__main__":
    app.run(main)
