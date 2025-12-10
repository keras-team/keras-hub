"""Convert DINOV3 checkpoints.

export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx

python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_small_lvd1689m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_small_lvd1689m
python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_small_plus_lvd1689m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_small_plus_lvd1689m
python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_base_lvd1689m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_base_lvd1689m
python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_large_lvd1689m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_large_lvd1689m
python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_huge_lvd1689m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_huge_lvd1689m
python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_huge_plus_lvd1689m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_huge_plus_lvd1689m
python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_7b_lvd1689m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_7b_lvd1689m

python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_large_sat493m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_large_sat493m
python tools/checkpoint_conversion/convert_dinov3_checkpoints.py \
    --preset dinov3_vit_7b_sat493m --upload_uri kaggle://kerashub/dinov3/keras/dinov3_vit_7b_sat493m
"""

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoModel

import keras_hub

PRESET_MAP = {
    # ViT lvd1689m variants.
    "dinov3_vit_small_lvd1689m": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vit_small_plus_lvd1689m": (
        "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    ),
    "dinov3_vit_base_lvd1689m": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vit_large_lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vit_huge_plus_lvd1689m": (
        "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    ),
    "dinov3_vit_7b_lvd1689m": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    # ViT sat493m variants.
    "dinov3_vit_large_sat493m": "facebook/dinov3-vitl16-pretrain-sat493m",
    "dinov3_vit_7b_sat493m": "facebook/dinov3-vit7b16-pretrain-sat493m",
}

FLAGS = flags.FLAGS
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


def convert_image_converter(image_size, hf_image_processor):
    config = hf_image_processor.to_dict()
    image_size = (image_size, image_size)
    std = config["image_std"]
    mean = config["image_mean"]
    return keras_hub.layers.DINOV3ImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        crop_to_aspect_ratio=False,
        interpolation="bilinear",
        antialias=True,
    )


def validate_output(
    keras_hub_model,
    keras_hub_image_converter,
    hf_model,
    hf_image_processor,
):
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file)

    # Preprocess with hf.
    hf_inputs = hf_image_processor(images=image, return_tensors="pt")
    hf_preprocessed = hf_inputs["pixel_values"].detach().cpu().numpy()

    # Preprocess with keras.
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
    images = keras_hub_image_converter(images)
    keras_preprocessed = keras.ops.convert_to_numpy(images)

    print("🔶 Keras preprocessor output:", keras_preprocessed[0, 0, :10, 0])
    print("🔶 HF preprocessor output:", hf_preprocessed[0, 0, 0, :10])

    # Call with hf. Use the keras preprocessed image so we can keep modeling
    # and preprocessing comparisons independent.
    hf_inputs["pixel_values"] = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_preprocessed, (0, 3, 1, 2))
        )
    )
    hf_outputs = hf_model(**hf_inputs)
    hf_outputs = hf_outputs[0].detach().cpu().numpy()

    # Call with keras.
    keras_outputs = keras_hub_model.predict({"pixel_values": images}, verbose=0)
    keras_outputs = keras.ops.convert_to_numpy(keras_outputs)

    print("🔶 Keras output:", keras_outputs[0, 0, :10])
    print("🔶 HF output:", hf_outputs[0, 0, :10])
    modeling_diff = np.mean(np.abs(keras_outputs - hf_outputs))
    print("🔶 Modeling difference:", modeling_diff)
    preprocessing_diff = np.mean(
        np.abs(keras_preprocessed - np.transpose(hf_preprocessed, (0, 2, 3, 1)))
    )
    print("🔶 Preprocessing difference:", preprocessing_diff)


def main(_):
    # === Get the preset name ===
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]

    # Load the HF model.
    hf_model = AutoModel.from_pretrained(hf_preset)
    hf_model.eval()
    image_size = int(hf_model.config.image_size)
    hf_image_processor = AutoImageProcessor.from_pretrained(hf_preset)

    # Load the KerasHub model.
    keras_hub_backbone = keras_hub.models.DINOV3Backbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_hub_backbone.summary()
    keras_hub_image_converter = convert_image_converter(
        image_size, hf_image_processor
    )
    print("✅ KerasHub model loaded.")
    print("✅ Weights converted.")

    validate_output(
        keras_hub_backbone,
        keras_hub_image_converter,
        hf_model,
        hf_image_processor,
    )
    print("✅ Output validated.")

    keras_hub_backbone.save_to_preset(f"./{preset}")
    keras_hub_image_converter.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
