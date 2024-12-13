"""
Convert PaliGemma2 checkpoints to the Keras format.

The checkpoints are from here:
https://www.kaggle.com/models/google/paligemma-2

The `vocabulary.spm` is from here:
https://www.kaggle.com/models/keras/paligemma/

The official repo is here:
https://github.com/google-research/big_vision

Setup:

```shell
git clone --quiet --branch=main --depth=1 \
    git@github.com:google-research/big_vision.git

pip install kaggle
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
```

Usage:

```shell
python -m tools.checkpoint_conversion.convert_pali_gemma2_checkpoints \
    --preset pali_gemma2_3b_pt_224
python -m tools.checkpoint_conversion.convert_pali_gemma2_checkpoints \
    --preset pali_gemma2_3b_pt_224 --weights_path ./path/to/weights.npz
python -m tools.checkpoint_conversion.convert_pali_gemma2_checkpoints \
    --preset pali_gemma2_3b_pt_224 --proto_path ./path/to/vocabulary.spm
python -m tools.checkpoint_conversion.convert_pali_gemma2_checkpoints \
    --preset pali_gemma2_3b_pt_224 \
    --upload_uri kaggle://keras/paligemma2/keras/pali_gemma2_3b_pt_224
```
"""

import functools
import io
import os
import pathlib
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
# No GPU for conversion, makes memory management easier.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax  # noqa: E402
import kagglehub  # noqa: E402
import keras  # noqa: E402
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402
import PIL  # noqa: E402
import requests  # noqa: E402
from absl import app  # noqa: E402
from absl import flags  # noqa: E402

import keras_hub  # noqa: E402

FLAGS = flags.FLAGS

PRESET_MAP = {
    "pali_gemma2_3b_ft_docci_448": (
        "google/paligemma-2/jax/paligemma2-3b-ft-docci-448"
    ),
    "pali_gemma2_10b_ft_docci_448": (
        "google/paligemma-2/jax/paligemma2-10b-ft-docci-448"
    ),
    "pali_gemma2_3b_pt_224": "google/paligemma-2/jax/paligemma2-3b-pt-224",
    "pali_gemma2_3b_pt_448": "google/paligemma-2/jax/paligemma2-3b-pt-448",
    "pali_gemma2_3b_pt_896": "google/paligemma-2/jax/paligemma2-3b-pt-896",
    "pali_gemma2_10b_pt_224": "google/paligemma-2/jax/paligemma2-10b-pt-224",
    "pali_gemma2_10b_pt_448": "google/paligemma-2/jax/paligemma2-10b-pt-448",
    "pali_gemma2_10b_pt_896": "google/paligemma-2/jax/paligemma2-10b-pt-896",
    "pali_gemma2_28b_pt_224": "google/paligemma-2/jax/paligemma2-28b-pt-224",
    "pali_gemma2_28b_pt_448": "google/paligemma-2/jax/paligemma2-28b-pt-448",
    "pali_gemma2_28b_pt_896": "google/paligemma-2/jax/paligemma2-28b-pt-896",
}


flags.DEFINE_string(
    "preset",
    None,
    f'Must be one of {",".join(PRESET_MAP.keys())}',
    required=True,
)
flags.DEFINE_string(
    "weights_path",
    None,
    "Optional path for the model weights to convert.",
)
flags.DEFINE_string(
    "proto_path",
    "vocabulary.spm",
    "Optional path for the SentencePiece proto file of the tokenizer.",
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def format_weights(weights):
    def recover_dtype(a):
        """Numpy's stores bfloat16 type as "void" type, so we recover it."""
        if hasattr(a, "dtype") and a.dtype.type is np.void:
            assert a.itemsize == 2, "Unknown dtype!"
            return a.view(ml_dtypes.bfloat16)
        else:
            return a

    weights = dict(weights)
    weights = jax.tree.map(recover_dtype, weights)

    formatted = {}

    # LLM part
    prefix = "params/llm"
    num_layers = int(weights[f"{prefix}/layers/mlp/linear"].shape[0])
    formatted["llm/embedding"] = weights[f"{prefix}/embedder/input_embedding"]
    for i in range(num_layers):
        layer_prefix = f"{prefix}/layers"
        formatted_prefix = f"llm/decoder_block_{i}"
        # RMSNorm
        formatted[f"{formatted_prefix}/pre_norm/scale"] = weights[
            f"{layer_prefix}/pre_attention_norm/scale"
        ][i]
        formatted[f"{formatted_prefix}/post_norm/scale"] = weights[
            f"{layer_prefix}/post_attention_norm/scale"
        ][i]
        formatted[f"{formatted_prefix}/pre_ffw_norm/scale"] = weights[
            f"{layer_prefix}/pre_ffw_norm/scale"
        ][i]
        formatted[f"{formatted_prefix}/post_ffw_norm/scale"] = weights[
            f"{layer_prefix}/post_ffw_norm/scale"
        ][i]
        # MHA
        formatted[f"{formatted_prefix}/mha/q/kernel"] = weights[
            f"{layer_prefix}/attn/q_einsum/w"
        ][i]
        formatted[f"{formatted_prefix}/mha/k/kernel"] = weights[
            f"{layer_prefix}/attn/kv_einsum/w"
        ][i, 0]
        formatted[f"{formatted_prefix}/mha/v/kernel"] = weights[
            f"{layer_prefix}/attn/kv_einsum/w"
        ][i, 1]
        formatted[f"{formatted_prefix}/mha/o/kernel"] = weights[
            f"{layer_prefix}/attn/attn_vec_einsum/w"
        ][i]
        # MLP
        formatted[f"{formatted_prefix}/ffw_gating/kernel"] = weights[
            f"{layer_prefix}/mlp/gating_einsum"
        ][i, 0]
        formatted[f"{formatted_prefix}/ffw_gating_2/kernel"] = weights[
            f"{layer_prefix}/mlp/gating_einsum"
        ][i, 1]
        formatted[f"{formatted_prefix}/ffw_linear/kernel"] = weights[
            f"{layer_prefix}/mlp/linear"
        ][i]
    formatted["llm/final_normalization/scale"] = weights[
        f"{prefix}/final_norm/scale"
    ]

    # ViT part
    prefix = "params/img"
    num_layers = int(
        weights[f"{prefix}/Transformer/encoderblock/LayerNorm_1/scale"].shape[0]
    )
    formatted["img/embedding/kernel"] = weights[f"{prefix}/embedding/kernel"]
    formatted["img/embedding/bias"] = weights[f"{prefix}/embedding/bias"]
    formatted["img/embedding/pos"] = weights[f"{prefix}/pos_embedding"]
    formatted["img/ln/gamma"] = weights[
        f"{prefix}/Transformer/encoder_norm/scale"
    ]
    formatted["img/ln/beta"] = weights[
        f"{prefix}/Transformer/encoder_norm/bias"
    ]
    for i in range(num_layers):
        encoder_prefix = f"{prefix}/Transformer/encoderblock"
        formatted_prefix = f"img/encoder_block_{i}"
        # MHA
        formatted[f"{formatted_prefix}/mha/q/kernel"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/query/kernel"
        ][i]
        formatted[f"{formatted_prefix}/mha/q/bias"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/query/bias"
        ][i]
        formatted[f"{formatted_prefix}/mha/k/kernel"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/key/kernel"
        ][i]
        formatted[f"{formatted_prefix}/mha/k/bias"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/key/bias"
        ][i]
        formatted[f"{formatted_prefix}/mha/v/kernel"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/value/kernel"
        ][i]
        formatted[f"{formatted_prefix}/mha/v/bias"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/value/bias"
        ][i]
        formatted[f"{formatted_prefix}/mha/o/kernel"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/out/kernel"
        ][i]
        formatted[f"{formatted_prefix}/mha/o/bias"] = weights[
            f"{encoder_prefix}/MultiHeadDotProductAttention_0/out/bias"
        ][i]
        # LN 0
        formatted[f"{formatted_prefix}/ln_0/gamma"] = weights[
            f"{encoder_prefix}/LayerNorm_0/scale"
        ][i]
        formatted[f"{formatted_prefix}/ln_0/beta"] = weights[
            f"{encoder_prefix}/LayerNorm_0/bias"
        ][i]
        # MLP
        formatted[f"{formatted_prefix}/mlp_1/kernel"] = weights[
            f"{encoder_prefix}/MlpBlock_0/Dense_0/kernel"
        ][i]
        formatted[f"{formatted_prefix}/mlp_1/bias"] = weights[
            f"{encoder_prefix}/MlpBlock_0/Dense_0/bias"
        ][i]
        formatted[f"{formatted_prefix}/mlp_2/kernel"] = weights[
            f"{encoder_prefix}/MlpBlock_0/Dense_1/kernel"
        ][i]
        formatted[f"{formatted_prefix}/mlp_2/bias"] = weights[
            f"{encoder_prefix}/MlpBlock_0/Dense_1/bias"
        ][i]
        # LN 1
        formatted[f"{formatted_prefix}/ln_1/gamma"] = weights[
            f"{encoder_prefix}/LayerNorm_1/scale"
        ][i]
        formatted[f"{formatted_prefix}/ln_1/beta"] = weights[
            f"{encoder_prefix}/LayerNorm_1/bias"
        ][i]
    formatted["img/head/kernel"] = weights[f"{prefix}/head/kernel"]
    formatted["img/head/bias"] = weights[f"{prefix}/head/bias"]
    return formatted


def convert_tokenizer(proto_path):
    try:
        tokenizer = keras_hub.models.PaliGemmaTokenizer(proto=proto_path)
    except Exception:
        raise FileNotFoundError(
            f"There is no proto file at proto_path={proto_path}. You can "
            "download it from https://www.kaggle.com/models/keras/paligemma/"
        )
    return tokenizer


def convert_image_converter(image_size):
    return keras_hub.layers.PaliGemmaImageConverter(
        image_size=(image_size, image_size),
        scale=1.0 / 127.5,
        offset=-1,
    )


def convert_model(preset):
    model_config = {
        "vocabulary_size": 257152,
        "vit_patch_size": 14,
        "vit_num_heads": 16,
        "vit_hidden_dim": 1152,
        "vit_num_layers": 27,
        # Gemma2
        "query_head_dim_normalize": True,
        "use_post_ffw_norm": True,
        "use_post_attention_norm": True,
        "final_logit_soft_cap": 30,
        "attention_logit_soft_cap": 50,
        "use_sliding_window_attention": True,
        "sliding_window_size": 4096,
    }
    preset = str(preset)

    # 2B, 10B, 28B
    if "_3b_" in preset:
        model_config.update(
            {
                "num_layers": 26,
                "num_query_heads": 8,
                "num_key_value_heads": 4,
                "hidden_dim": 2304,
                "intermediate_dim": 18432,
                "head_dim": 256,
            }
        )
    elif "_10b_" in preset:
        model_config.update(
            {
                "num_layers": 42,
                "num_query_heads": 16,
                "num_key_value_heads": 8,
                "hidden_dim": 3584,
                "intermediate_dim": 28672,
                "head_dim": 256,
            }
        )
    elif "_28b_" in preset:
        model_config.update(
            {
                "num_layers": 46,
                "num_query_heads": 32,
                "num_key_value_heads": 16,
                "hidden_dim": 4608,
                "intermediate_dim": 73728,
                "head_dim": 128,
                "query_head_dim_normalize": False,  # Only for 28B
            }
        )

    # Image size
    image_size = int(preset.split("_")[-1])
    model_config.update({"image_size": image_size})

    return keras_hub.models.PaliGemmaBackbone(**model_config)


def convert_weights(keras_model, weights):
    from keras_hub.src.models.pali_gemma.pali_gemma_decoder_block import (
        PaliGemmaDecoderBlock,
    )
    from keras_hub.src.models.pali_gemma.pali_gemma_vit import (
        PaliGemmaVitEncoder,
    )
    from keras_hub.src.models.pali_gemma.pali_gemma_vit import (
        PaliGemmaVitEncoderBlock,
    )

    if not isinstance(keras_model, keras_hub.models.PaliGemmaBackbone):
        raise ValueError(
            "`keras_model` must be a `keras_hub.models.PaliGemmaBackbone`. "
            f"Received: keras_model={keras_model} of type {type(keras_model)}"
        )

    # LLM part
    keras_model.token_embedding.embeddings.assign(weights["llm/embedding"])
    for i, layer in enumerate(keras_model.transformer_layers):
        if not isinstance(layer, PaliGemmaDecoderBlock):
            raise ValueError
        prefix = f"llm/decoder_block_{i}"
        # RMSNorm
        layer.pre_attention_norm.scale.assign(
            weights[f"{prefix}/pre_norm/scale"]
        )
        layer.post_attention_norm.scale.assign(
            weights[f"{prefix}/post_norm/scale"]
        )
        layer.pre_ffw_norm.scale.assign(weights[f"{prefix}/pre_ffw_norm/scale"])
        layer.post_ffw_norm.scale.assign(
            weights[f"{prefix}/post_ffw_norm/scale"]
        )
        # MHA
        layer.attention.query_dense.kernel.assign(
            weights[f"{prefix}/mha/q/kernel"]
        )
        layer.attention.key_dense.kernel.assign(
            weights[f"{prefix}/mha/k/kernel"]
        )
        layer.attention.value_dense.kernel.assign(
            weights[f"{prefix}/mha/v/kernel"]
        )
        layer.attention.output_dense.kernel.assign(
            weights[f"{prefix}/mha/o/kernel"]
        )
        # MLP
        layer.gating_ffw.kernel.assign(weights[f"{prefix}/ffw_gating/kernel"])
        layer.gating_ffw_2.kernel.assign(
            weights[f"{prefix}/ffw_gating_2/kernel"]
        )
        layer.ffw_linear.kernel.assign(weights[f"{prefix}/ffw_linear/kernel"])
    keras_model.layer_norm.scale.assign(
        weights["llm/final_normalization/scale"]
    )

    # ViT part
    vit_encoder = keras_model.vit_encoder.get_layer("image_encoder")
    if not isinstance(vit_encoder, PaliGemmaVitEncoder):
        raise ValueError
    vit_encoder.encoder_layer_norm.gamma.assign(weights["img/ln/gamma"])
    vit_encoder.encoder_layer_norm.beta.assign(weights["img/ln/beta"])
    vit_encoder.vision_embeddings.patch_embedding.kernel.assign(
        weights["img/embedding/kernel"]
    )
    vit_encoder.vision_embeddings.patch_embedding.bias.assign(
        weights["img/embedding/bias"]
    )
    vit_encoder.vision_embeddings.position_embedding.embeddings.assign(
        weights["img/embedding/pos"][0]
    )
    for i, layer in enumerate(vit_encoder.resblocks):
        if not isinstance(layer, PaliGemmaVitEncoderBlock):
            raise ValueError
        prefix = f"img/encoder_block_{i}"
        input_dim = hidden_dim = layer.attn.hidden_dim
        # MHA
        layer.attn.query_proj.kernel.assign(
            np.reshape(
                weights[f"{prefix}/mha/q/kernel"], (input_dim, hidden_dim)
            )
        )
        layer.attn.query_proj.bias.assign(
            np.reshape(weights[f"{prefix}/mha/q/bias"], (-1,))
        )
        layer.attn.key_proj.kernel.assign(
            np.reshape(
                weights[f"{prefix}/mha/k/kernel"], (input_dim, hidden_dim)
            )
        )
        layer.attn.key_proj.bias.assign(
            np.reshape(weights[f"{prefix}/mha/k/bias"], (-1,))
        )
        layer.attn.value_proj.kernel.assign(
            np.reshape(
                weights[f"{prefix}/mha/v/kernel"], (input_dim, hidden_dim)
            )
        )
        layer.attn.value_proj.bias.assign(
            np.reshape(weights[f"{prefix}/mha/v/bias"], (-1,))
        )
        layer.attn.out_proj.kernel.assign(
            np.reshape(
                weights[f"{prefix}/mha/o/kernel"], (input_dim, hidden_dim)
            )
        )
        layer.attn.out_proj.bias.assign(weights[f"{prefix}/mha/o/bias"])
        # LN 0
        layer.layer_norm_1.gamma.assign(weights[f"{prefix}/ln_0/gamma"])
        layer.layer_norm_1.beta.assign(weights[f"{prefix}/ln_0/beta"])
        # MLP
        layer.mlp_dense_1.kernel.assign(weights[f"{prefix}/mlp_1/kernel"])
        layer.mlp_dense_1.bias.assign(weights[f"{prefix}/mlp_1/bias"])
        layer.mlp_dense_2.kernel.assign(weights[f"{prefix}/mlp_2/kernel"])
        layer.mlp_dense_2.bias.assign(weights[f"{prefix}/mlp_2/bias"])
        # LN 1
        layer.layer_norm_2.gamma.assign(weights[f"{prefix}/ln_1/gamma"])
        layer.layer_norm_2.beta.assign(weights[f"{prefix}/ln_1/beta"])
    vit_classifier = keras_model.vit_encoder.get_layer("image_classifier")
    if not isinstance(vit_classifier, keras.layers.Dense):
        raise ValueError
    vit_classifier.kernel.assign(weights["img/head/kernel"])
    vit_classifier.bias.assign(weights["img/head/bias"])

    return keras_model


def validate_output(
    preset,
    keras_model,
    keras_tokenizer,
    keras_image_converter,
    big_vision_weights_path,
):
    def read_image(url):
        contents = io.BytesIO(requests.get(url).content)
        image = PIL.Image.open(contents)
        image = np.array(image).astype("float32")
        # Remove alpha channel if neccessary.
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image

    image = read_image(
        "https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png"
    )
    prompt = "describe en\n"
    max_length = 32
    preprocessor = keras_hub.models.PaliGemmaCausalLMPreprocessor(
        tokenizer=keras_tokenizer, image_converter=keras_image_converter
    )
    pali_gemma_lm = keras_hub.models.PaliGemmaCausalLM(
        preprocessor=preprocessor, backbone=keras_model
    )
    keras_output = pali_gemma_lm.generate(
        inputs={"images": image, "prompts": prompt}, max_length=max_length
    )
    keras_output = str(keras_output).replace(prompt, "")
    print("🔶 Prompt:", prompt.replace("\n", ""))
    print("🔶 KerasHub output:", keras_output)

    try:
        # Try using the official `big_vision` repo to validate the decoded
        # output.
        sys.path.append("big_vision")

        import jax.numpy as jnp
        import ml_collections
        from big_vision.models.proj.paligemma import paligemma
        from big_vision.trainers.proj.paligemma import predict_fns

        variant = preset.split("_")[2].lower()  # 3b, 10b, 28b
        if "b" not in variant:
            raise ValueError("🔶 Failed to parse the variant from the `preset`")
        gemma2_variant_mapping = {"3b": "2b", "10b": "9b", "28b": "27b"}
        big_vision_config = ml_collections.FrozenConfigDict(
            {
                "llm": {
                    "variant": f"gemma2_{gemma2_variant_mapping.get(variant)}",
                    "vocab_size": 257_152,
                },
                "img": {
                    "variant": "So400m/14",
                    "pool_type": "none",
                    "scan": True,
                    "dtype_mm": "bfloat16",
                },
            }
        )

        big_vision_model = paligemma.Model(**big_vision_config)
        big_vision_params = paligemma.load(
            None, str(big_vision_weights_path), big_vision_config
        )
        decode_fn = predict_fns.get_all(big_vision_model)["decode"]
        decode = functools.partial(
            decode_fn,
            devices=jax.devices(),
            eos_token=preprocessor.tokenizer.end_token_id,
        )

        preprocessed = preprocessor.generate_preprocess(
            {"images": image, "prompts": prompt}, sequence_length=max_length
        )
        images = jnp.expand_dims(preprocessed["images"], axis=0)
        token_ids = jnp.expand_dims(preprocessed["token_ids"], axis=0)
        big_vision_output = decode(
            {"params": big_vision_params},
            {
                "image": images,
                "text": token_ids,
                "mask_input": jnp.greater(token_ids, 0).astype("int32"),
                "mask_ar": jnp.zeros_like(token_ids),
                "_mask": jnp.array(True),
            },
            max_decode_len=max_length,
        )
        big_vision_output = big_vision_output[0]
        big_vision_output = preprocessor.generate_postprocess(
            {
                "token_ids": big_vision_output,
                "padding_mask": jnp.ones_like(big_vision_output).astype("bool"),
            }
        )
        print("🔶 big_vision output:", big_vision_output)
    except Exception as e:
        print(f"🔶 big_vision could not be run. Error: {e}")


def main(_):
    preset = str(FLAGS.preset)
    print(f"🏃 Coverting {preset}")

    # Currently all weights are bfloat16 (and have much faster download times
    # for it). We follow suit with Keras weights.
    keras.config.set_floatx("bfloat16")

    if FLAGS.weights_path is not None:
        big_vision_weights_path = pathlib.Path(FLAGS.weights_path)
    else:
        presets = PRESET_MAP.keys()
        if preset not in presets:
            raise ValueError(
                f"Invalid preset {preset}. Must be one of {list(presets)}"
            )
        handle = PRESET_MAP[preset]
        model_dir = kagglehub.model_download(handle)
        print("✅ JAX model downloaded from kaggle")

        files = list(pathlib.Path(model_dir).glob("*.npz"))
        if len(files) != 1:
            raise ValueError(
                f"Found too many files in {model_dir}. Expected only one file. "
                f"Recevied: {files}"
            )
        big_vision_weights_path = files[0]

    weights = np.load(big_vision_weights_path, allow_pickle=False)
    weights = format_weights(weights)
    image_size = int(preset.split("_")[-1])
    print("✅ JAX model weights loaded")

    keras_tokenizer = convert_tokenizer(FLAGS.proto_path)
    keras_image_converter = convert_image_converter(image_size)
    keras_model = convert_model(preset)
    print("✅ Keras model loaded")

    convert_weights(keras_model, weights)
    del weights
    print("✅ Weights converted")

    validate_output(
        preset,
        keras_model,
        keras_tokenizer,
        keras_image_converter,
        big_vision_weights_path,
    )
    print("✅ Output validated")

    keras_model.save_to_preset(preset)
    keras_tokenizer.save_to_preset(preset)
    keras_image_converter.save_to_preset(preset)
    del keras_model
    del keras_tokenizer
    del keras_image_converter
    print(f"🏁 Preset saved to ./{preset}")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
