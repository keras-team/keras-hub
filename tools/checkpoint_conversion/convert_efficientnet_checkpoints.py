"""
Convert efficientnet checkpoints.

python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b0_ra_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b0_ra_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b0_ra4_e3600_r224_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b0_ra4_e3600_r224_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b1_ft_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b1_ft_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b1_ra4_e3600_r240_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b1_ra4_e3600_r240_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b2_ra_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b2_ra_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b3_ra2_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b3_ra2_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b4_ra2_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b4_ra2_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b5_sw_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b5_sw_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_b5_sw_ft_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_b5_sw_ft_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_lite0_ra_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_lite0_ra_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_el_ra_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_el_ra_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_em_ra2_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_em_ra2_imagenet
python tools/checkpoint_conversion/convert_efficientnet_checkpoints.py \
    --preset efficientnet_es_ra_imagenet --upload_uri kaggle://keras/efficientnet/keras/efficientnet_es_ra_imagenet
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
    "efficientnet_b0_ra_imagenet": "timm/efficientnet_b0.ra_in1k",
    "efficientnet_b0_ra4_e3600_r224_imagenet": "timm/efficientnet_b0.ra4_e3600_r224_in1k",
    "efficientnet_b1_ft_imagenet": "timm/efficientnet_b1.ft_in1k",
    "efficientnet_b1_ra4_e3600_r240_imagenet": "timm/efficientnet_b1.ra4_e3600_r240_in1k",
    "efficientnet_b2_ra_imagenet": "timm/efficientnet_b2.ra_in1k",
    "efficientnet_b3_ra2_imagenet": "timm/efficientnet_b3.ra2_in1k",
    "efficientnet_b4_ra2_imagenet": "timm/efficientnet_b4.ra2_in1k",
    "efficientnet_b5_sw_imagenet": "timm/efficientnet_b5.sw_in12k",
    "efficientnet_b5_sw_ft_imagenet": "timm/efficientnet_b5.sw_in12k_ft_in1k",
    "efficientnet_lite0_ra_imagenet": "timm/efficientnet_lite0.ra_in1k",
    "efficientnet_el_ra_imagenet": "timm/efficientnet_el.ra_in1k",
    "efficientnet_em_ra2_imagenet": "timm/efficientnet_em.ra2_in1k",
    "efficientnet_es_ra_imagenet": "timm/efficientnet_es.ra_in1k",
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
    'Could be "kaggle://keras/{variant}/keras/{preset}_imagenet"',
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

    # Preprocess with Timm.
    data_config = timm.data.resolve_model_data_config(timm_model)
    data_config["crop_pct"] = 1.0  # Stop timm from cropping.
    transforms = timm.data.create_transform(**data_config, is_training=False)
    timm_preprocessed = transforms(image)
    timm_preprocessed = keras.ops.transpose(timm_preprocessed, axes=(1, 2, 0))
    timm_preprocessed = keras.ops.expand_dims(timm_preprocessed, 0)
    # Preprocess with Keras.
    batch = keras.ops.cast(batch, "float32")
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

    validate_output(keras_model, timm_model)

    keras_model.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
