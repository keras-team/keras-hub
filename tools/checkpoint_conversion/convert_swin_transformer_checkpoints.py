"""Convert Swin Transformer checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_swin_transformer_checkpoints.py \
    --preset swin_tiny_patch4_window7_224
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import AutoImageProcessor
from transformers import SwinModel

import keras_hub
from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)
from keras_hub.src.models.swin_transformer.swin_transformer_image_classifier_preprocessor import (  # noqa: E501
    SwinTransformerImageClassifierPreprocessor,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "swin_tiny_patch4_window7_224": "microsoft/swin-tiny-patch4-window7-224",
    "swin_small_patch4_window7_224": (
        "microsoft/swin-small-patch4-window7-224"
    ),
    "swin_base_patch4_window7_224": "microsoft/swin-base-patch4-window7-224",
    "swin_base_patch4_window12_384": (
        "microsoft/swin-base-patch4-window12-384"
    ),
    "swin_large_patch4_window7_224": (
        "microsoft/swin-large-patch4-window7-224"
    ),
    "swin_large_patch4_window12_384": (
        "microsoft/swin-large-patch4-window12-384"
    ),
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
    'Could be "kaggle://keras/swin_transformer/keras/{preset}"',
    required=False,
)


def validate_output(
    keras_backbone, keras_image_converter, hf_model, hf_processor
):
    """Validate converted model outputs match HuggingFace."""
    # Compare number of parameters between Keras and HF backbone.
    keras_params = keras_backbone.count_params()
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"🔶 Keras model params: {keras_params:,}")
    print(f"🔶 HF model params:    {hf_params:,}")
    assert keras_params == hf_params, (
        "❌ Parameter count mismatch between Keras and HF models!"
    )

    file = keras.utils.get_file(
        origin="http://images.cocodataset.org/val2017/000000039769.jpg"
    )
    image = Image.open(file)

    # Preprocess with Keras converter.
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
    keras_preprocessed = keras_image_converter(images)

    # Feed the same preprocessed pixels to HF (NCHW) to isolate modeling diff.
    hf_inputs = hf_processor(image, return_tensors="pt")
    hf_inputs["pixel_values"] = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_preprocessed, (0, 3, 1, 2))
        )
    )
    with torch.no_grad():
        hf_features = (
            hf_model(**hf_inputs).last_hidden_state.detach().cpu().numpy()
        )

    keras_features = keras.ops.convert_to_numpy(
        keras_backbone(keras_preprocessed, training=False)
    )

    print("🔶 Keras output (first token, first 10 dims):")
    print(f"   {keras_features[0, 0, :10]}")
    print("🔶 HF output (first token, first 10 dims):")
    print(f"   {hf_features[0, 0, :10]}")

    modeling_diff = np.mean(np.abs(keras_features - hf_features))
    max_diff = np.max(np.abs(keras_features - hf_features))
    relative_error = modeling_diff / np.mean(np.abs(hf_features))
    print(f"🔶 Modeling difference (mean): {modeling_diff:.6f}")
    print(f"🔶 Modeling difference (max):  {max_diff:.6f}")
    print(f"🔶 Relative error:             {relative_error * 100:.4f}%")

    # Also validate preprocessing matches HF.
    hf_preprocessed = (
        hf_processor(image, return_tensors="pt")["pixel_values"]
        .detach()
        .cpu()
        .numpy()
    )
    preprocessing_diff = np.mean(
        np.abs(
            keras.ops.convert_to_numpy(keras_preprocessed)
            - np.transpose(hf_preprocessed, (0, 2, 3, 1))
        )
    )
    print(f"🔶 Preprocessing difference:   {preprocessing_diff:.6f}")


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

    print(f"🏃 Converting {preset}")

    # Load HuggingFace model and processor.
    hf_model = SwinModel.from_pretrained(hf_preset)
    hf_processor = AutoImageProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    # Load backbone via on-the-fly conversion (uses convert_swin_transformer.py
    # under the hood).
    keras_backbone = SwinTransformerBackbone.from_preset(f"hf://{hf_preset}")
    print("✅ KerasHub backbone loaded via on-the-fly conversion.")
    print(f"   Parameters: {keras_backbone.count_params():,}")

    keras_preprocessor = SwinTransformerImageClassifierPreprocessor.from_preset(
        f"hf://{hf_preset}"
    )
    keras_image_converter = keras_preprocessor.image_converter
    # Keep resize target aligned with converter image size.
    image_height, image_width = keras_image_converter.image_size
    hf_processor.size = {"height": image_height, "width": image_width}

    validate_output(
        keras_backbone, keras_image_converter, hf_model, hf_processor
    )
    print("✅ Output validated.")

    keras_backbone.save_to_preset(f"./{preset}")
    keras_preprocessor.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
