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
from transformers import SwinForImageClassification

import keras_hub
from keras_hub.src.models.swin_transformer.swin_transformer_image_classifier import (  # noqa: E501
    SwinTransformerImageClassifier,
)
from keras_hub.src.utils.transformers import convert_swin_transformer

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


class StateDictLoader:
    """Minimal loader adapter to port weights from a HF PyTorch state_dict."""

    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.prefix = None

    def get_tensor(self, hf_weight_key):
        if hf_weight_key in self.state_dict:
            return self.state_dict[hf_weight_key]

        if self.prefix is not None:
            full_key = self.prefix + hf_weight_key
            if full_key in self.state_dict:
                return self.state_dict[full_key]

        for full_key in self.state_dict:
            if full_key.endswith(hf_weight_key) and full_key != hf_weight_key:
                self.prefix = full_key[: -len(hf_weight_key)]
                return self.state_dict[full_key]

        raise KeyError(f"Missing key in HF state_dict: {hf_weight_key}")

    def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
        hf_tensor = self.get_tensor(hf_weight_key)
        if hook_fn:
            hf_tensor = hook_fn(hf_tensor, list(keras_variable.shape))
        keras_variable.assign(hf_tensor)


flags.DEFINE_string(
    "upload_uri",
    None,
    'Could be "kaggle://keras/swin_transformer/keras/{preset}"',
    required=False,
)


def validate_output(keras_model, keras_image_converter, hf_model, hf_processor):
    """Validate converted classifier outputs match HuggingFace."""
    # Compare number of parameters between Keras and HF classifier.
    keras_params = keras_model.count_params()
    hf_params = hf_model.num_parameters()
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
        hf_logits = hf_model(**hf_inputs).logits.detach().cpu().numpy()

    keras_logits = keras.ops.convert_to_numpy(
        keras_model(keras_preprocessed, training=False)
    )
    keras_label = np.argmax(keras_logits[0])
    hf_label = np.argmax(hf_logits[0])

    print("🔶 Keras output (first 10 logits):")
    print(f"   {keras_logits[0, :10]}")
    print("🔶 HF output (first 10 logits):")
    print(f"   {hf_logits[0, :10]}")
    print(f"🔶 Keras label: {keras_label}")
    print(f"🔶 HF label:    {hf_label}")

    modeling_diff = np.mean(np.abs(keras_logits - hf_logits))
    max_diff = np.max(np.abs(keras_logits - hf_logits))
    relative_error = modeling_diff / np.mean(np.abs(hf_logits))
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
    hf_model = SwinForImageClassification.from_pretrained(hf_preset)
    hf_processor = AutoImageProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    # Load preset architecture/config only and port HF weights manually.
    # This avoids requiring model.safetensors for presets that only publish
    # pytorch_model.bin.
    keras_model = SwinTransformerImageClassifier.from_preset(
        f"hf://{hf_preset}",
        load_weights=False,
    )
    hf_state_dict = {
        key: value.detach().cpu().numpy()
        for key, value in hf_model.state_dict().items()
    }
    loader = StateDictLoader(hf_state_dict)
    hf_config = hf_model.config.to_dict()
    convert_swin_transformer.convert_weights(
        keras_model.backbone,
        loader,
        hf_config,
    )
    convert_swin_transformer.convert_head(
        keras_model,
        loader,
        hf_config,
    )
    print("✅ KerasHub classifier loaded via on-the-fly conversion.")
    print(f"   Parameters: {keras_model.count_params():,}")

    keras_preprocessor = keras_model.preprocessor
    keras_image_converter = keras_preprocessor.image_converter
    # Keep resize target aligned with converter image size.
    image_height, image_width = keras_image_converter.image_size
    hf_processor.size = {"height": image_height, "width": image_width}

    validate_output(keras_model, keras_image_converter, hf_model, hf_processor)
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
