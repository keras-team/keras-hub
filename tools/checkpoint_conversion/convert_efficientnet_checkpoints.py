# Copyright 2024 The KerasHub Authors
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
"""TODO: update example(s) for efficientnet

Convert resnet checkpoints.

python tools/checkpoint_conversion/convert_resnet_checkpoints.py \
    --preset resnet_18_imagenet --upload_uri kaggle://kerashub/resnetv1/keras/resnet_18_imagenet
python tools/checkpoint_conversion/convert_resnet_checkpoints.py \
    --preset resnet_50_imagenet --upload_uri kaggle://kerashub/resnetv1/keras/resnet_50_imagenet
python tools/checkpoint_conversion/convert_resnet_checkpoints.py \
    --preset resnet_101_imagenet --upload_uri kaggle://kerashub/resnetv1/keras/resnet_101_imagenet
python tools/checkpoint_conversion/convert_resnet_checkpoints.py \
    --preset resnet_152_imagenet --upload_uri kaggle://kerashub/resnetv1/keras/resnet_152_imagenet
python tools/checkpoint_conversion/convert_resnet_checkpoints.py \
    --preset resnet_v2_50_imagenet --upload_uri kaggle://kerashub/resnetv2/keras/resnet_v2_50_imagenet
python tools/checkpoint_conversion/convert_resnet_checkpoints.py \
    --preset resnet_v2_101_imagenet --upload_uri kaggle://kerashub/resnetv2/keras/resnet_v2_101_imagenet
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
    "enet_b0_ra": "timm/efficientnet_b0.ra_in1k",
    "enet_b1_ft": "timm/efficientnet_b1.ft_in1k",
    "enet_b1_pruned": "timm/efficientnet_b1_pruned.in1k",
    "enet_b2_ra": "timm/efficientnet_b2.ra_in1k",
    "enet_b2_pruned": "timm/efficientnet_b2_pruned.in1k",
    "enet_b3_ra2": "timm/efficientnet_b3.ra2_in1k",
    "enet_b3_pruned": "timm/efficientnet_b3_pruned.in1k",
    "enet_b4_ra2": "timm/efficientnet_b4.ra2_in1k",
    "enet_b5_sw": "timm/efficientnet_b5.sw_in12k",
    "enet_b5_sw_ft": "timm/efficientnet_b5.sw_in12k_ft_in1k",
    "enet_el_ra": "timm/efficientnet_el.ra_in1k",
    "enet_el_pruned": "timm/efficientnet_el_pruned.in1k",
    "enet_em_ra2": "timm/efficientnet_em.ra2_in1k",
    "enet_es_ra": "timm/efficientnet_es.ra_in1k",
    "enet_es_pruned": "timm/efficientnet_es_pruned.in1k",
    "enet_b0_ra4_e3600_r224": "timm/efficientnet_b0.ra4_e3600_r224_in1k",
    "enet_b1_ra4_e3600_r240": "timm/efficientnet_b1.ra4_e3600_r240_in1k",
    "enet2_rw_m_agc": "timm/efficientnetv2_rw_m.agc_in1k",
    "enet2_rw_s_ra2": "timm/efficientnetv2_rw_s.ra2_in1k",
    "enet2_rw_t_ra2": "timm/efficientnetv2_rw_t.ra2_in1k",
}
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "preset",
    None,
    "Must be a valid `EfficientNet` preset from KerasHub",
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

    # Call with Timm.
    timm_batch = keras_model.preprocessor(batch)
    timm_batch = keras.ops.transpose(timm_batch, axes=(0, 3, 1, 2)) / 255.0
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
    # import ipdb; ipdb.set_trace()
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
