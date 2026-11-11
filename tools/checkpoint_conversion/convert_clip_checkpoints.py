"""Convert CLIP checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_base_patch16 --upload_uri kaggle://kerashub/clip/keras/clip_vit_base_patch16
python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_base_patch32 --upload_uri kaggle://kerashub/clip/keras/clip_vit_base_patch32
python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_large_patch14 --upload_uri kaggle://kerashub/clip/keras/clip_vit_large_patch14
python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_large_patch14_336 --upload_uri kaggle://kerashub/clip/keras/clip_vit_large_patch14_336
python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_b_32_laion2b_s34b_b79k --upload_uri kaggle://kerashub/clip/keras/clip_vit_b_32_laion2b_s34b_b79k
python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_h_14_laion2b_s32b_b79k --upload_uri kaggle://kerashub/clip/keras/clip_vit_h_14_laion2b_s32b_b79k
python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_g_14_laion2b_s12b_b42k --upload_uri kaggle://kerashub/clip/keras/clip_vit_g_14_laion2b_s12b_b42k
python tools/checkpoint_conversion/convert_clip_checkpoints.py \
    --preset clip_vit_bigg_14_laion2b_39b_b160k --upload_uri kaggle://kerashub/clip/keras/clip_vit_bigg_14_laion2b_39b_b160k
"""

import os
import shutil
import tempfile

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import CLIPModel
from transformers import CLIPProcessor

import keras_hub
from keras_hub.src.models.clip.clip_backbone import CLIPBackbone
from keras_hub.src.models.clip.clip_image_converter import CLIPImageConverter
from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer

FLAGS = flags.FLAGS

PRESET_MAP = {
    "clip_vit_base_patch16": "openai/clip-vit-base-patch16",
    "clip_vit_base_patch32": "openai/clip-vit-base-patch32",
    "clip_vit_large_patch14": "openai/clip-vit-large-patch14",
    "clip_vit_large_patch14_336": "openai/clip-vit-large-patch14-336",
    "clip_vit_b_32_laion2b_s34b_b79k": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "clip_vit_h_14_laion2b_s32b_b79k": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "clip_vit_g_14_laion2b_s12b_b42k": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    # No config.json
    # "clip_vit_g_14_laion2b_s34b_b88k": (
    #     "laion/CLIP-ViT-g-14-laion2B-s34B-b88K"
    # ),
    "clip_vit_bigg_14_laion2b_39b_b160k": (
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
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
    'Could be "kaggle://keras/{variant}/keras/{preset}"',
    required=False,
)


def load_keras_hub_components(hf_model, hf_preset, hf_dir):
    """Load KerasHub CLIP components via the transformers converter.

    Weight porting lives in `keras_hub/src/utils/transformers/convert_clip.py`
    and is shared with `from_preset("hf://...")`. Some upstream checkpoints
    only ship `pytorch_model.bin` (or sharded safetensors without an index),
    which the safetensors-based converter cannot read directly. To support all
    presets, we re-save the HF model as a local HF-format directory with
    safetensors and load the backbone from there. The tokenizer and image
    converter only need JSON/text files, which every upstream repo provides,
    so they are loaded from the Hub directly.
    """
    hf_model.save_pretrained(hf_dir, safe_serialization=True)
    keras_model = CLIPBackbone.from_preset(hf_dir)
    hf_uri = f"hf://{hf_preset}"
    keras_image_converter = CLIPImageConverter.from_preset(hf_uri)
    keras_tokenizer = CLIPTokenizer.from_preset(hf_uri)
    return keras_model, keras_image_converter, keras_tokenizer


def validate_output(
    keras_model,
    keras_image_converter,
    keras_tokenizer,
    hf_model,
    hf_model_processor,
):
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file)
    text = ["a photo of a cat", "a photo of a dog"]

    # Preprocess with hf.
    hf_inputs = hf_model_processor(
        text=text,
        images=[image, image],
        return_tensors="pt",
        padding="max_length",
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
    hf_vision_logits = hf_outputs.logits_per_image.detach().cpu().numpy()

    # Call with keras.
    keras_preprocessor = CLIPPreprocessor(keras_tokenizer)
    token_ids = keras_preprocessor(
        {"images": keras_preprocessed, "prompts": text}
    )["token_ids"]
    keras_outputs = keras_model.predict(
        {"images": images, "token_ids": token_ids}, verbose=0
    )
    keras_vision_logits = keras.ops.convert_to_numpy(
        keras_outputs["vision_logits"]
    )

    print("🔶 Keras output:", keras_vision_logits[0, :10])
    print("🔶 HF output:", hf_vision_logits[0, :10])
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
    hf_model = CLIPModel.from_pretrained(hf_preset)
    hf_preprocessor = CLIPProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    # Load KerasHub components through the transformers converter
    # (`keras_hub/src/utils/transformers/convert_clip.py`).
    with tempfile.TemporaryDirectory() as hf_dir:
        keras_model, keras_image_converter, keras_tokenizer = (
            load_keras_hub_components(hf_model, hf_preset, hf_dir)
        )
    print("✅ KerasHub model loaded and weights converted.")

    validate_output(
        keras_model,
        keras_image_converter,
        keras_tokenizer,
        hf_model,
        hf_preprocessor,
    )
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    keras_image_converter.save_to_preset(f"./{preset}")
    keras_tokenizer.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
