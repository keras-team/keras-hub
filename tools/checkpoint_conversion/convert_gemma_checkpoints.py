"""
Convert Gemma flax checkpoints to the Keras format.

The flax checkpoint should match the directory structure here:
https://www.kaggle.com/models/google/gemma/flax

The flax directory should have a sentenepiece proto, and an inner directory with
an orbax checkpoint:
tokenizer.model
2b-it/_METADATA
2b-it/checkpoint
2b-it/...

Setup:
```shell
pip install -r requirements.txt
pip install git+https://github.com/google-deepmind/gemma.git
python pip_build.py --install
```

Usage:
```shell
cd tools/checkpoint_conversion
python convert_gemma_checkpoints.py --preset gemma_2b_en
python convert_gemma_checkpoints.py --preset new_gemma --flax_dir ./new_gemma
```
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

import keras_hub  # noqa: E402

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
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)

flags.DEFINE_string(
    "flax_dir",
    None,
    "Optional path to a local flax directory to convert. See the script "
    "docstring for more details on the format of the flax directory.",
)


def download_flax_model(handle):
    return kagglehub.model_download(handle)


def convert_model(flax_config, flax_params, vocab_size):
    kwargs = {}
    # Hack to infer Gemma 2 config options until Flax actually adds support.
    if "post_attention_norm" in flax_params["transformer"]["layer_0"]:
        # The 27B parameter model is the only model that does a weird
        # query normalization.
        is_gemma2_27b = flax_config.num_heads == 32
        # We would like to convert these from Flax, but have no way until
        # flax supports Gemma 2.
        kwargs = {
            "query_head_dim_normalize": not is_gemma2_27b,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "final_logit_soft_cap": 30,
            "attention_logit_soft_cap": 50,
            "use_sliding_window_attention": True,
            "sliding_window_size": 4096,
        }
    return keras_hub.models.GemmaBackbone(
        vocabulary_size=vocab_size,
        num_layers=flax_config.num_layers,
        num_query_heads=flax_config.num_heads,
        num_key_value_heads=flax_config.num_kv_heads,
        hidden_dim=flax_config.embed_dim,
        intermediate_dim=flax_config.hidden_dim * 2,
        head_dim=flax_config.head_dim,
        **kwargs,
    )


def convert_tokenizer(proto_path):
    return keras_hub.models.GemmaTokenizer(proto=proto_path)


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

        if "post_attention_norm" in flax_block:
            keras_block.post_attention_norm.set_weights(
                [flax_block["post_attention_norm"]["scale"]]
            )
        if "post_ffw_norm" in flax_block:
            keras_block.post_ffw_norm.set_weights(
                [flax_block["post_ffw_norm"]["scale"]]
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

    # KerasHub
    preprocessor = keras_hub.models.GemmaCausalLMPreprocessor(keras_tokenizer)
    gemma_lm = keras_hub.models.GemmaCausalLM(
        backbone=keras_model,
        preprocessor=preprocessor,
    )
    keras_output = gemma_lm.generate([input_str], max_length=length)
    keras_output = keras_output[0]
    print("🔶 KerasHub output:", keras_output)

    # Flax
    try:
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
            # Length of "<bos>What is Keras?"
            total_generation_steps=length - 5,
        )
        flax_output = input_str + flax_output.text[0]
        print("🔶 Flax output:", flax_output)
    except Exception as e:
        print("🔶 Flax could not be run.", e)


def main(_):
    preset = FLAGS.preset

    print(f"🏃 Coverting {preset}")

    # Currently all flax weights are bfloat16 (and have much faster download
    # times for it). We follow suit with Keras weights.
    keras.config.set_floatx("bfloat16")

    if FLAGS.flax_dir is not None:
        flax_dir = FLAGS.flax_dir
    else:
        presets = PRESET_MAP.keys()
        assert (
            preset in presets
        ), f"Invalid preset {preset}. Must be one of {','.join(presets)}"
        handle = PRESET_MAP[preset]
        flax_dir = download_flax_model(handle)

    proto_path = flax_dir + "/tokenizer.model"
    print("✅ Flax model downloaded from kaggle")

    checkpoint_dir = None
    for path in os.listdir(flax_dir):
        checkpoint_file = os.path.join(flax_dir, path, "_METADATA")
        if os.path.exists(checkpoint_file):
            checkpoint_dir = os.path.join(flax_dir, path)
    assert checkpoint_dir is not None, "Cannot find orbax checkpoint files"

    flax_tokenier = sentencepiece.SentencePieceProcessor()
    flax_tokenier.Load(proto_path)
    flax_params = params_lib.load_and_format_params(checkpoint_dir)
    flax_config = transformer_lib.TransformerConfig.from_params(flax_params)
    print("✅ Flax model loaded")

    keras_tokenizer = convert_tokenizer(proto_path)
    vocab_size = keras_tokenizer.vocabulary_size()
    keras_model = convert_model(flax_config, flax_params, vocab_size)
    print("✅ Keras model loaded")

    convert_weights(keras_model, flax_config, flax_params)
    print("✅ Weights converted")

    validate_output(keras_model, keras_tokenizer, flax_params, flax_tokenier)
    print("✅ Output validated")

    keras_model.save_to_preset(preset)
    keras_tokenizer.save_to_preset(preset)
    print(f"🏁 Preset saved to ./{preset}")


if __name__ == "__main__":
    app.run(main)
