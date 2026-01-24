"""Convert DINOV2 checkpoints.

export KAGGLE_USERNAME=xxx KAGGLE_KEY=xxx

python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_small --upload_uri kaggle://kerashub/dinov2/keras/dinov2_small
python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_base --upload_uri kaggle://kerashub/dinov2/keras/dinov2_base
python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_large --upload_uri kaggle://kerashub/dinov2/keras/dinov2_large
python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_giant --upload_uri kaggle://kerashub/dinov2/keras/dinov2_giant

python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_with_registers_small --upload_uri kaggle://kerashub/dinov2/keras/dinov2_with_registers_small
python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_with_registers_base --upload_uri kaggle://kerashub/dinov2/keras/dinov2_with_registers_base
python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_with_registers_large --upload_uri kaggle://kerashub/dinov2/keras/dinov2_with_registers_large
python tools/checkpoint_conversion/convert_dinov2_checkpoints.py \
    --preset dinov2_with_registers_giant --upload_uri kaggle://kerashub/dinov2/keras/dinov2_with_registers_giant
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
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base": "facebook/dinov2-base",
    "dinov2_large": "facebook/dinov2-large",
    "dinov2_giant": "facebook/dinov2-giant",
    "dinov2_with_registers_small": "facebook/dinov2-with-registers-small",
    "dinov2_with_registers_base": "facebook/dinov2-with-registers-base",
    "dinov2_with_registers_large": "facebook/dinov2-with-registers-large",
    "dinov2_with_registers_giant": "facebook/dinov2-with-registers-giant",
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
    return keras_hub.layers.DINOV2ImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        interpolation="bicubic",  # DINOV2 defaults to bicubic resampling.
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
    keras_outputs = keras_hub_model.predict({"images": images}, verbose=0)
    keras_outputs = keras.ops.convert_to_numpy(keras_outputs)

    print("🔶 Keras output:", keras_outputs[0])
    print("🔶 HF output:", hf_outputs[0])
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
    hf_image_processor = AutoImageProcessor.from_pretrained(
        hf_preset,
        # We don't perform shortest edge resizing and center cropping in
        # KerasHub.
        size={"shortest_edge": image_size},
        crop_size={"height": image_size, "width": image_size},
    )

    # Load the KerasHub model.
    keras_hub_backbone = keras_hub.models.DINOV2Backbone.from_preset(
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
