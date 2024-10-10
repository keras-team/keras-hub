"""Loads an external VGG model and saves it in Keras format.

Optionally uploads the model to Keras if the `--upload_uri` flag is passed.

python tools/checkpoint_conversion/convert_vgg_checkpoints.py \
    --preset vgg11 --upload_uri kaggle://kerashub/vgg/keras/vgg11
"""

import os
import shutil

import keras
import numpy as np
import PIL
import timm
import torch
from absl import app
from absl import flags

import keras_hub

PRESET_MAP = {
    "vgg11": "timm/vgg11.tv_in1k",
    "vgg13": "timm/vgg13.tv_in1k",
    "vgg16": "timm/vgg16.tv_in1k",
    "vgg19": "timm/vgg19.tv_in1k",
    # TODO(jeffcarp): Add BN variants.
}


PRESET = flags.DEFINE_string(
    "preset",
    None,
    "Must be a valid `VGG` preset from KerasHub",
    required=True,
)
UPLOAD_URI = flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}_int8"',
)


def validate_output(keras_model, timm_model):
    file = keras.utils.get_file(
        origin=(
            "https://storage.googleapis.com/keras-cv/"
            "models/paligemma/cow_beach_1.png"
        )
    )
    image = PIL.Image.open(file)
    batch = np.array([image])

    # Preprocess with Timm.
    data_config = timm.data.resolve_model_data_config(timm_model)
    data_config["crop_pct"] = 1.0  # Stop timm from cropping.
    transforms = timm.data.create_transform(**data_config, is_training=False)
    timm_preprocessed = transforms(image)
    timm_preprocessed = keras.ops.transpose(timm_preprocessed, axes=(1, 2, 0))
    timm_preprocessed = keras.ops.expand_dims(timm_preprocessed, 0)

    # Preprocess with Keras.
    keras_preprocessed = keras_model.preprocessor(batch)

    # Call with Timm. Use the keras preprocessed image so we can keep modeling
    # and preprocessing comparisons independent.
    timm_batch = keras.ops.transpose(keras_preprocessed, axes=(0, 3, 1, 2))
    timm_batch = torch.from_numpy(np.array(timm_batch))
    timm_outputs = timm_model(timm_batch).detach().numpy()
    timm_label = np.argmax(timm_outputs[0])

    # Call with Keras.
    keras_outputs = keras_model.predict(batch)
    keras_label = np.argmax(keras_outputs[0])

    print("🔶 Keras output:", keras_outputs[0, :10])
    print("🔶 TIMM output:", timm_outputs[0, :10])
    print("🔶 Keras label:", keras_label)
    print("🔶 TIMM label:", timm_label)
    modeling_diff = np.mean(np.abs(keras_outputs - timm_outputs))
    print("🔶 Modeling difference:", modeling_diff)
    preprocessing_diff = np.mean(np.abs(keras_preprocessed - timm_preprocessed))
    print("🔶 Preprocessing difference:", preprocessing_diff)


def main(_):
    preset = PRESET.value
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    timm_name = PRESET_MAP[preset]

    timm_model = timm.create_model(timm_name, pretrained=True)
    timm_model = timm_model.eval()
    print("✅ Loaded TIMM model.")
    print(timm_model)

    keras_model = keras_hub.models.ImageClassifier.from_preset(
        "hf://" + timm_name,
    )
    print("✅ Loaded KerasHub model.")

    keras_model.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}")

    validate_output(keras_model, timm_model)

    upload_uri = UPLOAD_URI.value
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
