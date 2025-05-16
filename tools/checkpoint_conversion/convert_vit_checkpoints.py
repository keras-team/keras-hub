"""Convert ViT checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_vit_checkpoints.py \
    --preset vit_base_patch16_224
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import ViTForImageClassification
from transformers import ViTImageProcessor

import keras_hub
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.models.vit.vit_image_classifier import ViTImageClassifier
from keras_hub.src.models.vit.vit_image_classifier_preprocessor import (
    ViTImageClassifierPreprocessor,
)
from keras_hub.src.models.vit.vit_image_converter import ViTImageConverter

FLAGS = flags.FLAGS

PRESET_MAP = {
    "vit_base_patch16_224": "google/vit-base-patch16-224",
    "vit_base_patch16_384": "google/vit-base-patch16-384",
    "vit_base_patch32_384": "google/vit-base-patch32-384",
    "vit_large_patch16_224": "google/vit-large-patch16-224",
    "vit_large_patch16_384": "google/vit-large-patch16-384",
    "vit_large_patch32_384": "google/vit-large-patch32-384",
    "vit_base_patch16_224_in21k": "google/vit-base-patch16-224-in21k",
    "vit_base_patch32_224_in21k": "google/vit-base-patch32-224-in21k",
    "vit_large_patch16_224_in21k": "google/vit-large-patch16-224-in21k",
    "vit_large_patch32_224_in21k": "google/vit-large-patch32-224-in21k",
    "vit_huge_patch14_224_in21k": "google/vit-huge-patch14-224-in21k",
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


def convert_model(hf_model):
    config = hf_model.config.to_dict()
    image_size = config["image_size"]
    backbone = ViTBackbone(
        image_shape=(image_size, image_size, 3),
        patch_size=(config["patch_size"], config["patch_size"]),
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        hidden_dim=config["hidden_size"],
        mlp_dim=config["intermediate_size"],
        dropout_rate=config["hidden_dropout_prob"],
        attention_dropout=config["attention_probs_dropout_prob"],
        use_mha_bias=config["qkv_bias"],
    )

    return backbone, config


def convert_backbone_weights(backbone, hf_model):
    state_dict = hf_model.state_dict()
    state_dict.update(hf_model.named_buffers())

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
            f"{weight_key}.attention.query.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.query_dense.bias,
            f"{weight_key}.attention.query.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # key
        port_weights(
            keras_variable.key_dense.kernel,
            f"{weight_key}.attention.key.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.key_dense.bias,
            f"{weight_key}.attention.key.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # value
        port_weights(
            keras_variable.value_dense.kernel,
            f"{weight_key}.attention.value.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (hidden_dim, num_heads, hidden_dim // num_heads)
            ),
        )
        port_weights(
            keras_variable.value_dense.bias,
            f"{weight_key}.attention.value.bias",
            hook_fn=lambda x, _: np.reshape(
                x, (num_heads, hidden_dim // num_heads)
            ),
        )
        # output
        port_weights(
            keras_variable.output_dense.kernel,
            f"{weight_key}.output.dense.weight",
            hook_fn=lambda x, _: np.reshape(
                x.T, (num_heads, hidden_dim // num_heads, hidden_dim)
            ),
        )
        port_weights(
            keras_variable.output_dense.bias, f"{weight_key}.output.dense.bias"
        )

    port_weights(
        backbone.layers[1].patch_embedding.kernel,
        "vit.embeddings.patch_embeddings.projection.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )

    port_weights(
        backbone.layers[1].patch_embedding.bias,
        "vit.embeddings.patch_embeddings.projection.bias",
    )

    port_weights(
        backbone.layers[1].class_token,
        "vit.embeddings.cls_token",
    )

    port_weights(
        backbone.layers[1].position_embedding.embeddings,
        "vit.embeddings.position_embeddings",
        hook_fn=lambda x, _: x[0],
    )
    encoder_layers = backbone.layers[2].encoder_layers
    for i, encoder_block in enumerate(encoder_layers):
        prefix = "vit.encoder.layer"
        num_heads = encoder_block.num_heads
        hidden_dim = encoder_block.hidden_dim

        port_mha(
            encoder_block.mha,
            f"{prefix}.{i}.attention",
            num_heads,
            hidden_dim,
        )
        port_ln(encoder_block.layer_norm_1, f"{prefix}.{i}.layernorm_before")
        port_ln(encoder_block.layer_norm_2, f"{prefix}.{i}.layernorm_after")

        port_dense(
            encoder_block.mlp.dense_1, f"{prefix}.{i}.intermediate.dense"
        )
        port_dense(encoder_block.mlp.dense_2, f"{prefix}.{i}.output.dense")

    port_ln(backbone.layers[2].layer_norm, "vit.layernorm")
    # port_dense(keras_hub_model.output_dense, "classifier")


def convert_head_weights(keras_model, hf_model):
    state_dict = hf_model.state_dict()
    state_dict.update(hf_model.named_buffers())

    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    prefix = "classifier."

    port_weights(
        keras_model.output_dense.kernel,
        prefix + "weight",
        hook_fn=lambda x, _: x.T,
    )
    port_weights(
        keras_model.output_dense.bias,
        prefix + "bias",
    )


def convert_image_converter(hf_image_processor):
    config = hf_image_processor.to_dict()
    image_size = (config["size"]["height"], config["size"]["width"])
    std = config["image_std"]
    mean = config["image_mean"]
    return ViTImageConverter(
        image_size=image_size,
        scale=[1.0 / config["rescale_factor"] / s for s in mean],
        offset=[-m / s for m, s in zip(mean, std)],
        interpolation="bilinear",  # ViT defaults to bilinear resampling.
    )


def validate_output(
    keras_model,
    keras_image_converter,
    hf_model,
    hf_image_processor,
    head_weights=False,
):
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file)

    # Preprocess with hf.
    hf_inputs = hf_image_processor(
        image,
        return_tensors="pt",
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
    if head_weights:
        hf_vision_logits = hf_outputs.logits.detach().cpu().numpy()

    else:
        hf_vision_logits = hf_outputs.last_hidden_state.detach().cpu().numpy()

    # Call with keras.
    keras_outputs = keras_model(keras_preprocessed)
    keras_vision_logits = keras.ops.convert_to_numpy(keras_outputs)

    print("🔶 Keras output:", keras_vision_logits[0, :10])
    print("🔶 HF output:", hf_vision_logits[0, :10])
    if head_weights:
        print(
            "🔶 HF top 5 ImageNet outputs:",
            keras_hub.utils.decode_imagenet_predictions(hf_vision_logits),
        )
        print(
            "🔶 Keras top 5 ImageNet outputs:",
            keras_hub.utils.decode_imagenet_predictions(keras_outputs),
        )
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

    # Load huggingface model.
    hf_model = ViTForImageClassification.from_pretrained(hf_preset)
    hf_preprocessor = ViTImageProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    keras_backbone, hf_config = convert_model(hf_model)
    keras_image_converter = convert_image_converter(hf_preprocessor)
    keras_image_preprocessor = ViTImageClassifierPreprocessor(
        image_converter=keras_image_converter
    )
    print("✅ KerasHub model loaded.")

    convert_backbone_weights(keras_backbone, hf_model)
    print("✅ Backbone weights converted.")

    if hf_config["architectures"][0] == "ViTForImageClassification":
        keras_model = ViTImageClassifier(
            backbone=keras_backbone, num_classes=len(hf_config["id2label"])
        )
        convert_head_weights(keras_model, hf_model)
        print("✅ Head weights converted.")
        validate_output(
            keras_model,
            keras_image_converter,
            hf_model,
            hf_preprocessor,
            head_weights=True,
        )
        print("✅ Output validated.")
        keras_model.preprocessor = keras_image_preprocessor
        keras_model.save_to_preset(f"./{preset}")
    else:
        hf_model = hf_model.vit
        validate_output(
            keras_backbone,
            keras_image_converter,
            hf_model,
            hf_preprocessor,
        )
        print("✅ Output validated.")
        keras_backbone.save_to_preset(f"./{preset}")
        keras_image_preprocessor.save_to_preset(f"./{preset}")

    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
