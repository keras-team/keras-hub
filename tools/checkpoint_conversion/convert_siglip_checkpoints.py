"""Convert SigLIP checkpoints.

export KAGGLE_USERNAME=xxx
export KAGGLE_KEY=xxx

python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_base_patch16_224 --upload_uri kaggle://kerashub/siglip/keras/siglip_base_patch16_224
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_base_patch16_256 --upload_uri kaggle://kerashub/siglip/keras/siglip_base_patch16_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_base_patch16_384 --upload_uri kaggle://kerashub/siglip/keras/siglip_base_patch16_384
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_base_patch16_512 --upload_uri kaggle://kerashub/siglip/keras/siglip_base_patch16_512
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_large_patch16_256 --upload_uri kaggle://kerashub/siglip/keras/siglip_large_patch16_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_large_patch16_384 --upload_uri kaggle://kerashub/siglip/keras/siglip_large_patch16_384
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_so400m_patch14_224 --upload_uri kaggle://kerashub/siglip/keras/siglip_so400m_patch14_224
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_so400m_patch14_384 --upload_uri kaggle://kerashub/siglip/keras/siglip_so400m_patch14_384
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_so400m_patch16_256_i18n --upload_uri kaggle://kerashub/siglip/keras/siglip_so400m_patch16_256_i18n
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip_base_patch16_256_multilingual --upload_uri kaggle://kerashub/siglip/keras/siglip_base_patch16_256_multilingual

python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_base_patch16_224 --upload_uri kaggle://kerashub/siglip/keras/siglip2_base_patch16_224
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_base_patch16_256 --upload_uri kaggle://kerashub/siglip/keras/siglip2_base_patch16_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_base_patch32_256 --upload_uri kaggle://kerashub/siglip/keras/siglip2_base_patch32_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_base_patch16_384 --upload_uri kaggle://kerashub/siglip/keras/siglip2_base_patch16_384
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_base_patch16_512 --upload_uri kaggle://kerashub/siglip/keras/siglip2_base_patch16_512
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_large_patch16_256 --upload_uri kaggle://kerashub/siglip/keras/siglip2_large_patch16_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_large_patch16_384 --upload_uri kaggle://kerashub/siglip/keras/siglip2_large_patch16_384
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_large_patch16_512 --upload_uri kaggle://kerashub/siglip/keras/siglip2_large_patch16_512
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_giant_opt_patch16_256 --upload_uri kaggle://kerashub/siglip/keras/siglip2_giant_opt_patch16_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_giant_opt_patch16_384 --upload_uri kaggle://kerashub/siglip/keras/siglip2_giant_opt_patch16_384
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_so400m_patch14_224 --upload_uri kaggle://kerashub/siglip/keras/siglip2_so400m_patch14_224
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_so400m_patch14_256 --upload_uri kaggle://kerashub/siglip/keras/siglip2_so400m_patch14_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_so400m_patch16_256 --upload_uri kaggle://kerashub/siglip/keras/siglip2_so400m_patch16_256
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_so400m_patch16_384 --upload_uri kaggle://kerashub/siglip/keras/siglip2_so400m_patch16_384
python tools/checkpoint_conversion/convert_siglip_checkpoints.py \
    --preset siglip2_so400m_patch16_512 --upload_uri kaggle://kerashub/siglip/keras/siglip2_so400m_patch16_512
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import SiglipModel
from transformers import SiglipProcessor

import keras_hub
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.models.siglip.siglip_backbone import SigLIPBackbone
from keras_hub.src.models.siglip.siglip_image_converter import (
    SigLIPImageConverter,
)
from keras_hub.src.models.siglip.siglip_preprocessor import SigLIPPreprocessor
from keras_hub.src.models.siglip.siglip_text_encoder import SigLIPTextEncoder
from keras_hub.src.models.siglip.siglip_tokenizer import SigLIPTokenizer
from keras_hub.src.models.siglip.siglip_vision_encoder import (
    SigLIPVisionEncoder,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "siglip_base_patch16_224": "google/siglip-base-patch16-224",
    "siglip_base_patch16_256": "google/siglip-base-patch16-256",
    "siglip_base_patch16_384": "google/siglip-base-patch16-384",
    "siglip_base_patch16_512": "google/siglip-base-patch16-512",
    "siglip_large_patch16_256": "google/siglip-large-patch16-256",
    "siglip_large_patch16_384": "google/siglip-large-patch16-384",
    "siglip_so400m_patch14_224": "google/siglip-so400m-patch14-224",
    "siglip_so400m_patch14_384": "google/siglip-so400m-patch14-384",
    "siglip_so400m_patch16_256_i18n": "google/siglip-so400m-patch16-256-i18n",
    "siglip_base_patch16_256_multilingual": "google/siglip-base-patch16-256-multilingual",  # noqa: E501
    # SigLIP2 (NaFlex version is not supported yet)
    "siglip2_base_patch16_224": "google/siglip2-base-patch16-224",
    "siglip2_base_patch16_256": "google/siglip2-base-patch16-256",
    "siglip2_base_patch32_256": "google/siglip2-base-patch32-256",
    "siglip2_base_patch16_384": "google/siglip2-base-patch16-384",
    "siglip2_base_patch16_512": "google/siglip2-base-patch16-512",
    "siglip2_large_patch16_256": "google/siglip2-large-patch16-256",
    "siglip2_large_patch16_384": "google/siglip2-large-patch16-384",
    "siglip2_large_patch16_512": "google/siglip2-large-patch16-512",
    "siglip2_giant_opt_patch16_256": "google/siglip2-giant-opt-patch16-256",
    "siglip2_giant_opt_patch16_384": "google/siglip2-giant-opt-patch16-384",
    "siglip2_so400m_patch14_224": "google/siglip2-so400m-patch14-224",
    "siglip2_so400m_patch14_384": "google/siglip2-so400m-patch14-384",
    "siglip2_so400m_patch16_256": "google/siglip2-so400m-patch16-256",
    "siglip2_so400m_patch16_384": "google/siglip2-so400m-patch16-384",
    "siglip2_so400m_patch16_512": "google/siglip2-so400m-patch16-512",
    "medsiglip_900m_448": "google/medsiglip-448",
}

flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {','.join(PRESET_MAP.keys())}",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def convert_model(hf_model, dtype=None):
    vision_encoder_config = hf_model.vision_model.config.to_dict()
    text_encoder_config = hf_model.text_model.config.to_dict()
    image_size = vision_encoder_config["image_size"]
    if vision_encoder_config["hidden_act"] == "gelu_pytorch_tanh":
        vision_encoder_config["hidden_act"] = "gelu_approximate"
    vision_encoder = SigLIPVisionEncoder(
        patch_size=vision_encoder_config["patch_size"],
        hidden_dim=vision_encoder_config["hidden_size"],
        num_layers=vision_encoder_config["num_hidden_layers"],
        num_heads=vision_encoder_config["num_attention_heads"],
        intermediate_dim=vision_encoder_config["intermediate_size"],
        intermediate_activation=vision_encoder_config["hidden_act"],
        layer_norm_epsilon=vision_encoder_config["layer_norm_eps"],
        image_shape=(image_size, image_size, 3),
        dtype=dtype,
    )
    if text_encoder_config["hidden_act"] == "gelu_pytorch_tanh":
        text_encoder_config["hidden_act"] = "gelu_approximate"
    text_encoder = SigLIPTextEncoder(
        vocabulary_size=text_encoder_config["vocab_size"],
        embedding_dim=text_encoder_config["hidden_size"],
        hidden_dim=text_encoder_config["hidden_size"],
        num_layers=text_encoder_config["num_hidden_layers"],
        num_heads=text_encoder_config["num_attention_heads"],
        intermediate_dim=text_encoder_config["intermediate_size"],
        intermediate_activation=text_encoder_config["hidden_act"],
        layer_norm_epsilon=text_encoder_config["layer_norm_eps"],
        max_sequence_length=text_encoder_config["max_position_embeddings"],
        projection_dim=text_encoder_config.get("projection_size"),
        dtype=dtype,
    )
    return SigLIPBackbone(vision_encoder, text_encoder, dtype=dtype)


def convert_weights(keras_hub_model, hf_model):
    # Get `state_dict` from `hf_model`.
    state_dict = hf_model.state_dict()
    state_dict.update(hf_model.named_buffers())  # Add buffers.

    # Helper functions.
    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    def port_ln(keras_variable, weight_key):
        port_weights(keras_variable.gamma, f"{weight_key}.weight")
        port_weights(keras_variable.beta, f"{weight_key}.bias")

    def port_dense(keras_variable, weight_key):
        port_weights(
            keras_variable.kernel,
            f"{weight_key}.weight",
            hook_fn=lambda x, _: x.T,
        )
        if keras_variable.bias is not None:
            port_weights(keras_variable.bias, f"{weight_key}.bias")

    def port_mha(keras_variable, weight_key, num_heads, hidden_dim):
        # query
        port_weights(
            keras_variable.query_dense.kernel,
            f"{weight_key}.q_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.query_dense.bias,
            f"{weight_key}.q_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        port_weights(
            keras_variable.key_dense.kernel,
            f"{weight_key}.k_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.key_dense.bias,
            f"{weight_key}.k_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # value
        port_weights(
            keras_variable.value_dense.kernel,
            f"{weight_key}.v_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.value_dense.bias,
            f"{weight_key}.v_proj.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        port_weights(
            keras_variable.output_dense.kernel,
            f"{weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        port_weights(
            keras_variable.output_dense.bias, f"{weight_key}.out_proj.bias"
        )

    def port_mha_torch(keras_variable, weight_key, num_heads, hidden_dim):
        # query
        port_weights(
            keras_variable.query_dense.kernel,
            f"{weight_key}.in_proj_weight",
            hook_fn=lambda x, _: np.reshape(
                x[:hidden_dim].T,
                (hidden_dim, num_heads, hidden_dim // num_heads),
            ),
        )
        port_weights(
            keras_variable.query_dense.bias,
            f"{weight_key}.in_proj_bias",
            hook_fn=lambda x, _: np.reshape(
                x[:hidden_dim], (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        port_weights(
            keras_variable.key_dense.kernel,
            f"{weight_key}.in_proj_weight",
            hook_fn=lambda x, _: np.reshape(
                x[hidden_dim : hidden_dim * 2].T,
                (hidden_dim, num_heads, hidden_dim // num_heads),
            ),
        )
        port_weights(
            keras_variable.key_dense.bias,
            f"{weight_key}.in_proj_bias",
            hook_fn=lambda x, _: np.reshape(
                x[hidden_dim : hidden_dim * 2],
                (num_heads, hidden_dim // num_heads),
            ),
        )
        # value
        port_weights(
            keras_variable.value_dense.kernel,
            f"{weight_key}.in_proj_weight",
            hook_fn=lambda x, _: np.reshape(
                x[hidden_dim * 2 :].T,
                (hidden_dim, num_heads, hidden_dim // num_heads),
            ),
        )
        port_weights(
            keras_variable.value_dense.bias,
            f"{weight_key}.in_proj_bias",
            hook_fn=lambda x, _: np.reshape(
                x[hidden_dim * 2 :], (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        port_weights(
            keras_variable.output_dense.kernel,
            f"{weight_key}.out_proj.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        port_weights(
            keras_variable.output_dense.bias, f"{weight_key}.out_proj.bias"
        )

    assert isinstance(keras_hub_model, SigLIPBackbone)

    # Port vision encoder.
    vision_encoder = keras_hub_model.vision_encoder
    assert isinstance(vision_encoder, SigLIPVisionEncoder)
    # Embedding
    port_weights(
        vision_encoder.embedding.position_ids,
        "vision_model.embeddings.position_ids",
    )
    port_weights(
        vision_encoder.embedding.patch_embedding.kernel,
        "vision_model.embeddings.patch_embedding.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    port_weights(
        vision_encoder.embedding.patch_embedding.bias,
        "vision_model.embeddings.patch_embedding.bias",
    )
    port_weights(
        vision_encoder.embedding.position_embedding.embeddings,
        "vision_model.embeddings.position_embedding.weight",
    )
    encoder_layers = vision_encoder.encoder_layers
    for i in range(len(encoder_layers)):
        prefix = "vision_model.encoder.layers"
        num_heads = encoder_layers[i].num_heads
        hidden_dim = encoder_layers[i].hidden_dim
        port_mha(
            encoder_layers[i].self_attn,
            f"{prefix}.{i}.self_attn",
            num_heads,
            hidden_dim,
        )
        port_ln(
            encoder_layers[i].layer_norm1,
            f"{prefix}.{i}.layer_norm1",
        )
        port_ln(
            encoder_layers[i].layer_norm2,
            f"{prefix}.{i}.layer_norm2",
        )
        port_dense(encoder_layers[i].mlp.fc1, f"{prefix}.{i}.mlp.fc1")
        port_dense(encoder_layers[i].mlp.fc2, f"{prefix}.{i}.mlp.fc2")
    # Post LN
    port_ln(vision_encoder.post_layer_norm, "vision_model.post_layernorm")
    # Head
    port_weights(vision_encoder.head.probe, "vision_model.head.probe")
    port_mha_torch(
        vision_encoder.head.attention,
        "vision_model.head.attention",
        vision_encoder.head.num_heads,
        vision_encoder.head.hidden_dim,
    )
    port_ln(vision_encoder.head.layer_norm, "vision_model.head.layernorm")
    port_dense(vision_encoder.head.mlp.fc1, "vision_model.head.mlp.fc1")
    port_dense(vision_encoder.head.mlp.fc2, "vision_model.head.mlp.fc2")

    # Port text encoder.
    text_encoder = keras_hub_model.text_encoder
    assert isinstance(text_encoder, SigLIPTextEncoder)
    # Embedding
    port_weights(
        text_encoder.embedding.token_embedding._embeddings,
        "text_model.embeddings.token_embedding.weight",
    )
    port_weights(
        text_encoder.embedding.position_embedding._embeddings,
        "text_model.embeddings.position_embedding.weight",
    )
    port_weights(
        text_encoder.embedding.position_ids,
        "text_model.embeddings.position_ids",
    )
    encoder_layers = text_encoder.encoder_layers
    for i in range(len(encoder_layers)):
        prefix = "text_model.encoder.layers"
        num_heads = encoder_layers[i].num_heads
        hidden_dim = encoder_layers[i].hidden_dim
        port_mha(
            encoder_layers[i].self_attn,
            f"{prefix}.{i}.self_attn",
            num_heads,
            hidden_dim,
        )
        port_ln(
            encoder_layers[i].layer_norm1,
            f"{prefix}.{i}.layer_norm1",
        )
        port_ln(
            encoder_layers[i].layer_norm2,
            f"{prefix}.{i}.layer_norm2",
        )
        port_dense(encoder_layers[i].mlp.fc1, f"{prefix}.{i}.mlp.fc1")
        port_dense(encoder_layers[i].mlp.fc2, f"{prefix}.{i}.mlp.fc2")
    # LN
    port_ln(text_encoder.post_layer_norm, "text_model.final_layer_norm")
    # Head
    port_dense(text_encoder.head, "text_model.head")

    # Port logit scale and bias.
    port_weights(
        keras_hub_model.siglip_head.logit_scale,
        "logit_scale",
        hook_fn=lambda x, _: x[0],
    )
    port_weights(
        keras_hub_model.siglip_head.logit_bias,
        "logit_bias",
        hook_fn=lambda x, _: x[0],
    )


def convert_image_converter(hf_image_processor):
    config = hf_image_processor.to_dict()
    image_size = (config["size"]["height"], config["size"]["width"])
    std = config["image_std"]
    mean = config["image_mean"]
    return SigLIPImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        interpolation="bicubic",  # SigLIP defaults to bicubic resampling.
        antialias=True,
        crop_to_aspect_ratio=False,
    )


def convert_tokenizer(hf_tokenizer, is_siglip2=False):
    if is_siglip2:
        return GemmaTokenizer(hf_tokenizer.vocab_file)
    else:
        return SigLIPTokenizer(hf_tokenizer.vocab_file, add_eos=True)


def validate_output(
    keras_model,
    keras_image_converter,
    keras_tokenizer,
    hf_model,
    hf_model_processor,
    is_siglip2=False,
):
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file)
    text = ["a photo of 2 cats", "a photo of 2 dogs"]

    # Preprocess with hf.
    hf_inputs = hf_model_processor(
        text=text,
        images=[image, image],
        return_tensors="pt",
        padding="max_length",
        max_length=64 if is_siglip2 else None,
    )
    hf_preprocessed = hf_inputs["pixel_values"].detach().cpu().numpy()

    # Preprocess with keras.
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
    images = np.concatenate([images, images], axis=0)
    images = keras_image_converter(images)
    keras_preprocessed = keras.ops.convert_to_numpy(images)

    # Call with hf. Use the keras preprocessed image so we can keep modeling
    # and preprocessing comparisons independent.
    hf_inputs["pixel_values"] = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_preprocessed, (0, 3, 1, 2))
        )
    )
    hf_outputs = hf_model(**hf_inputs)
    hf_vision_logits = hf_outputs.logits_per_image.detach().cpu().numpy()

    # Call with keras.
    keras_preprocessor = SigLIPPreprocessor(
        keras_tokenizer,
        sequence_length=(
            64 if is_siglip2 else hf_model_processor.tokenizer.model_max_length
        ),
    )
    token_ids = keras_preprocessor(
        {"images": keras.ops.convert_to_numpy(images), "prompts": text}
    )["token_ids"]
    keras_outputs = keras_model.predict(
        {"images": images, "token_ids": token_ids}, verbose=0
    )
    keras_vision_logits = keras.ops.convert_to_numpy(
        keras_outputs["vision_logits"]
    )

    print("🔶 Keras output:", keras_vision_logits[0])
    print("🔶 HF output:", hf_vision_logits[0])
    modeling_diff = np.mean(np.abs(keras_vision_logits - hf_vision_logits))
    print("🔶 Modeling difference:", modeling_diff)
    preprocessing_diff = np.mean(
        np.abs(keras_preprocessed - np.transpose(hf_preprocessed, (0, 2, 3, 1)))
    )
    print("🔶 Preprocessing difference:", preprocessing_diff)


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    print(f"🏃 Coverting {preset}")

    # Whether the model is SigLIP2.
    is_siglip2 = "siglip2" in preset

    # Load huggingface model.
    hf_model = SiglipModel.from_pretrained(hf_preset)
    hf_preprocessor = SiglipProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    keras_model = convert_model(hf_model)
    keras_model.summary()
    keras_image_converter = convert_image_converter(
        hf_preprocessor.image_processor
    )
    keras_tokenizer = convert_tokenizer(hf_preprocessor.tokenizer, is_siglip2)
    print("✅ KerasHub model loaded.")

    convert_weights(keras_model, hf_model)
    print("✅ Weights converted.")

    validate_output(
        keras_model,
        keras_image_converter,
        keras_tokenizer,
        hf_model,
        hf_preprocessor,
        is_siglip2,
    )
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    keras_image_converter.save_to_preset(f"./{preset}")
    keras_tokenizer.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)

if __name__ == "__main__":
    app.run(main)
