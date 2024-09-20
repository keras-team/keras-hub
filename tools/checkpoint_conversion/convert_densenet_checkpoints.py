# Copyright 2024 The KerasHUB Authors
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
"""Convert densenet checkpoints.

python tools/checkpoint_conversion/convert_densenet_checkpoints.py \
    --preset densenet_121_imagenet --upload_uri kaggle://kerashub/densenet/keras/densenet_121_imagenet
python tools/checkpoint_conversion/convert_densenet_checkpoints.py \
    --preset densenet_169_imagenet --upload_uri kaggle://kerashub/densenet/keras/densenet_169_imagenet
python tools/checkpoint_conversion/convert_densenet_checkpoints.py \
    --preset densenet_201_imagenet --upload_uri kaggle://kerashub/densenet/keras/densenet_201_imagenet
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
    "densenet_121_imagenet": "timm/densenet121.tv_in1k",
    "densenet_169_imagenet": "timm/densenet169.tv_in1k",
    "densenet_201_imagenet": "timm/densenet201.tv_in1k",
}
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "preset",
    None,
    "Must be a valid `DenseNet` preset from KerasHUB",
    required=True,
)
flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/{variant}/keras/{preset}_int8"',
    required=False,
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
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Call with Timm.
    timm_batch = keras_model.preprocessor(batch)
    timm_batch = keras.ops.transpose(timm_batch, axes=(0, 3, 1, 2)) / 255
    timm_batch = (timm_batch - mean) / std
    timm_batch = torch.from_numpy(np.array(timm_batch))
    timm_outputs = timm_model(timm_batch).detach().numpy()
    timm_label = np.argmax(timm_outputs[0])
    # Call with Keras.
    keras_outputs = keras_model.predict(batch)
    keras_label = np.argmax(keras_outputs[0])

    print("🔶 Keras output:", keras_outputs[0, :10])
    print("🔶 TIMM output:", timm_outputs[0, :10])
    print("🔶 Difference:", np.mean(np.abs(keras_outputs - timm_outputs)))
    print("🔶 Keras label:", keras_label)
    print("🔶 TIMM label:", timm_label)


def main(_):
    preset = FLAGS.preset
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    timm_name = PRESET_MAP[preset]

    print("✅ Loaded TIMM model.")
    timm_model = timm.create_model(timm_name, pretrained=True)
    timm_model = timm_model.eval()

    print("✅ Loaded KerasHub model.")
    keras_model = keras_hub.models.ImageClassifier.from_preset(
        "hf://" + timm_name,
    )

    keras_model.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}")

    validate_output(keras_model, timm_model)

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
