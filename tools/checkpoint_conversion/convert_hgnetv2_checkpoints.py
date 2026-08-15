"""Convert HGNetV2 checkpoints from Hugging Face to Keras.

Usage:
    export KAGGLE_USERNAME=XXX
    export KAGGLE_KEY=XXX

    python tools/checkpoint_conversion/convert_hgnetv2_checkpoints.py
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from timm import create_model

import keras_hub
from keras_hub.src.utils.imagenet.imagenet_utils import (
    decode_imagenet_predictions,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "hgnetv2_b6_ssld_stage2_ft_in1k": "timm/hgnetv2_b6.ssld_stage2_ft_in1k",
    "hgnetv2_b6_ssld_stage1_in22k_in1k": "timm/hgnetv2_b6.ssld_stage1_in22k_in1k",  # noqa: E501
    "hgnetv2_b5_ssld_stage2_ft_in1k": "timm/hgnetv2_b5.ssld_stage2_ft_in1k",
    "hgnetv2_b5_ssld_stage1_in22k_in1k": "timm/hgnetv2_b5.ssld_stage1_in22k_in1k",  # noqa: E501
    "hgnetv2_b4_ssld_stage2_ft_in1k": "timm/hgnetv2_b4.ssld_stage2_ft_in1k",
}

flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/hgnetv2/keras/{preset}"',
    required=False,
)


def validate_output(keras_model, hf_model):
    file = keras.utils.get_file(
        origin="https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg"
    )
    image = Image.open(file)
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
    keras_preprocessed = keras_model.preprocessor(images)
    keras_preprocessed = keras.ops.convert_to_tensor(
        keras_preprocessed, dtype="float32"
    )
    keras_logits = keras_model.predict(keras_preprocessed)

    hf_inputs = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_preprocessed, (0, 3, 1, 2))
        )
    )
    last_stage_name = keras_model.backbone.stage_names[-1]
    keras_backbone_output = keras_model.backbone(
        keras_preprocessed, training=False
    )[last_stage_name]
    hf_backbone = torch.nn.Sequential(*list(hf_model.children())[:-1])
    hf_backbone_output = hf_backbone(hf_inputs)

    keras_output_np = keras.ops.convert_to_numpy(keras_backbone_output)
    hf_output_np = np.transpose(
        hf_backbone_output.detach().cpu().numpy(), (0, 2, 3, 1)
    )
    print(
        "Keras top 5 ImageNet predictions:",
        decode_imagenet_predictions(keras_logits, top=5),
    )
    np.testing.assert_allclose(
        keras_output_np, hf_output_np, atol=1e-4, rtol=1e-4
    )
    print("Numerical check passed.")


def main(_):
    for preset, hf_preset in PRESET_MAP.items():
        if os.path.exists(preset):
            shutil.rmtree(preset)

        print(f"\nConverting {preset}")
        hf_model = create_model(hf_preset, pretrained=True)
        hf_model.eval()

        # `ImageClassifier.from_preset` dispatches through
        # `convert_hgnetv2.py`, which handles config translation and weight
        # porting directly from the Hugging Face safetensors checkpoint.
        keras_model = keras_hub.models.ImageClassifier.from_preset(
            f"hf://{hf_preset}"
        )
        print("KerasHub model loaded.")

        validate_output(keras_model, hf_model)

        keras_model.save_to_preset(f"./{preset}")
        print(f"Preset saved to ./{preset}.")

        upload_uri = FLAGS.upload_uri
        if upload_uri:
            keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
            print(f"Successfully uploaded {preset} to {upload_uri}")
    print("\nAll presets validated and saved successfully!")


if __name__ == "__main__":
    app.run(main)
