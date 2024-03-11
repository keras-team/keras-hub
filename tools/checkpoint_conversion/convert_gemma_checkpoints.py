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
Convert Gemma flax checkpoints to the Keras format.

Setup:
pip install -r requirements.txt
pip install git+https://github.com/google-deepmind/gemma.git
python pip_build.py --install

Usage:
cd tools/checkpoint_conversion
python convert_gemma_checkpoints.py --preset gemma_2b_en
"""

import os

os.environ["KERAS_BACKEND"] = "jax"
# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import kagglehub  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
import sentencepiece  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from gemma import params as params_lib  # noqa: E402
from gemma import sampler as sampler_lib  # noqa: E402
from gemma import transformer as transformer_lib  # noqa: E402

import keras_nlp  # noqa: E402

FLAGS = flags.FLAGS

PRESET_MAP = {
    "gemma_2b_en": "google/gemma/flax/2b",
    "gemma_7b_en": "google/gemma/flax/7b",
    "gemma_instruct_2b_en": "google/gemma/flax/2b-it",
    "gemma_instruct_7b_en": "google/gemma/flax/7b-it",
}


flags.DEFINE_string(
    "preset",
    None,
    f'Must be one of {",".join(PRESET_MAP.keys())}',
    required=True,
)


def download_flax_model(handle):
    return kagglehub.model_download(handle)


def convert_model(flax_config, vocab_size):
    return keras_nlp.models.GemmaBackbone(
        vocabulary_size=vocab_size,
        num_layers=flax_config.num_layers,
        num_query_heads=flax_config.num_heads,
        num_key_value_heads=flax_config.num_kv_heads,
        hidden_dim=flax_config.embed_dim,
        intermediate_dim=flax_config.hidden_dim * 2,
        head_dim=flax_config.head_dim,
    )


def convert_tokenizer(proto_path):
    return keras_nlp.models.GemmaTokenizer(proto=proto_path)


def convert_weights(keras_model, flax_config, flax_params):
    # Chomp the embedding weights. Upstream pads for TPU efficiency, but this
    # leads to weird gotchas (you need to disregard part of your output logits).
    embeddings = flax_params["transformer"]["embedder"]["input_embedding"]
    embeddings = np.asarray(embeddings[: keras_model.vocabulary_size, :])
    keras_model.get_layer("token_embedding").set_weights([embeddings])
    keras_model.get_layer("final_normalization").set_weights(
        [np.asarray(flax_params["transformer"]["final_norm"]["scale"])]
    )
    for i in range(flax_config.num_layers):
        flax_layer_name = f"layer_{i}"
        keras_block = keras_model.get_layer(f"decoder_block_{i}")

        flax_block = flax_params["transformer"][flax_layer_name]
        keras_block.pre_attention_norm.set_weights(
            [flax_block["pre_attention_norm"]["scale"]]
        )
        keras_block.pre_ffw_norm.set_weights(
            [flax_block["pre_ffw_norm"]["scale"]]
        )

        keras_block.gating_ffw.set_weights(
            [flax_block["mlp"]["gating_einsum"][0]]
        )
        keras_block.gating_ffw_2.set_weights(
            [flax_block["mlp"]["gating_einsum"][1]]
        )
        keras_block.ffw_linear.set_weights([flax_block["mlp"]["linear"]])

        attn_block = flax_block["attn"]
        if flax_config.num_heads != flax_config.num_kv_heads:
            # MQA.
            keras_block.attention.query_dense.kernel.assign(
                np.asarray(attn_block["q_einsum"]["w"][:, :, :])
            )
            keras_block.attention.key_dense.kernel.assign(
                np.asarray(attn_block["kv_einsum"]["w"][0, :, :, :])
            )
            keras_block.attention.value_dense.kernel.assign(
                np.asarray(attn_block["kv_einsum"]["w"][1, :, :, :])
            )
        else:
            # MHA.
            keras_block.attention.query_dense.kernel.assign(
                np.asarray(attn_block["qkv_einsum"]["w"][0, :, :, :])
            )
            keras_block.attention.key_dense.kernel.assign(
                np.asarray(attn_block["qkv_einsum"]["w"][1, :, :, :])
            )
            keras_block.attention.value_dense.kernel.assign(
                np.asarray(attn_block["qkv_einsum"]["w"][2, :, :, :])
            )
        keras_block.attention.output_dense.kernel.assign(
            flax_block["attn"]["attn_vec_einsum"]["w"]
        )


def validate_output(
    keras_model,
    keras_tokenizer,
    flax_params,
    flax_tokenizer,
):
    input_str = "What is Keras?"
    length = 32

    # KerasNLP
    preprocessor = keras_nlp.models.GemmaCausalLMPreprocessor(keras_tokenizer)
    gemma_lm = keras_nlp.models.GemmaCausalLM(
        backbone=keras_model,
        preprocessor=preprocessor,
    )
    keras_output = gemma_lm.generate([input_str], max_length=length)
    keras_output = keras_output[0]

    # Flax
    transformer_config = transformer_lib.TransformerConfig.from_params(
        flax_params,
        cache_size=length,
    )
    transformer = transformer_lib.Transformer(transformer_config)
    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=flax_tokenizer,
        params=flax_params["transformer"],
    )
    flax_output = sampler(
        input_strings=[input_str],
        total_generation_steps=length - 5,  # Length of "<bos>What is Keras?"
    )
    flax_output = input_str + flax_output.text[0]

    # Comparing the outputs.
    print("🔶 KerasNLP output:", keras_output)
    print("🔶 Flax output:", flax_output)


def main(_):
    preset = FLAGS.preset

    assert (
        preset in PRESET_MAP.keys()
    ), f'Invalid preset {preset}. Must be one of {",".join(PRESET_MAP.keys())}'

    print(f"🏃 Coverting {preset}")

    # Currently all flax weights are bfloat16 (and have much faster download
    # times for it). We follow suit with Keras weights.
    keras.config.set_floatx("bfloat16")

    handle = PRESET_MAP[preset]
    flax_dir = download_flax_model(handle)
    proto_path = flax_dir + "/tokenizer.model"
    print("✅ Flax model downloaded from kaggle")

    variant = handle.split("/")[-1]
    flax_tokenier = sentencepiece.SentencePieceProcessor()
    flax_tokenier.Load(proto_path)
    flax_params = params_lib.load_and_format_params(flax_dir + "/" + variant)
    flax_config = transformer_lib.TransformerConfig.from_params(flax_params)
    print("✅ Flax model loaded")

    keras_tokenizer = convert_tokenizer(proto_path)
    vocab_size = keras_tokenizer.vocabulary_size()
    keras_model = convert_model(flax_config, vocab_size)
    print("✅ Keras model loaded")

    convert_weights(keras_model, flax_config, flax_params)
    print("✅ Weights converted")

    validate_output(keras_model, keras_tokenizer, flax_params, flax_tokenier)
    print("✅ Output validated")

    keras_nlp.src.utils.preset_utils.save_to_preset(keras_model, preset)
    keras_nlp.src.utils.preset_utils.save_to_preset(
        keras_tokenizer, preset, config_filename="tokenizer.json"
    )
    print(f"🏁 Preset saved to ./{preset}")


if __name__ == "__main__":
    app.run(main)
