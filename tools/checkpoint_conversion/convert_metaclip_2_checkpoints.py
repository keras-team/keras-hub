"""Convert MetaCLIP 2 checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_huge_patch14_224 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_huge_patch14_224
python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_huge_patch14_378 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_huge_patch14_378
python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_giant_patch14_224 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_giant_patch14_224
python tools/checkpoint_conversion/convert_metaclip_2_checkpoints.py \
    --preset metaclip_2_vit_giant_patch14_378 --upload_uri kaggle://keras/metaclip_2/keras/metaclip_2_vit_giant_patch14_378
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers import AutoModel
from transformers import AutoProcessor

import keras_hub

FLAGS = flags.FLAGS

# MetaCLIP 2 Worldwide models from Meta AI (non-distilled)
# These correspond to the models from the Meta CLIP 2 collection:
# https://huggingface.co/collections/facebook/meta-clip-2
PRESET_MAP = {
    # ViT-H-14-quickgelu-worldwide (224 resolution)
    "metaclip_2_vit_huge_patch14_224": "facebook/metaclip-2-worldwide-huge-quickgelu",  # noqa
    # ViT-H-14-378-worldwide (378 resolution)
    "metaclip_2_vit_huge_patch14_378": "facebook/metaclip-2-worldwide-huge-378",
    # ViT-bigG-14-worldwide (224 resolution)
    "metaclip_2_vit_giant_patch14_224": "facebook/metaclip-2-worldwide-giant",
    # ViT-bigG-14-378-worldwide (378 resolution)
    "metaclip_2_vit_giant_patch14_378": "facebook/metaclip-2-worldwide-giant-378",  # noqa
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


def convert_image_converter(hf_image_processor):
    """Convert HuggingFace image processor to Keras Hub image converter."""
    config = hf_image_processor.to_dict()
    image_size = (config["crop_size"]["height"], config["crop_size"]["width"])
    std = config["image_std"]
    mean = config["image_mean"]
    return keras_hub.layers.MetaCLIP2ImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        interpolation="bicubic",  # MetaCLIP 2 defaults to bicubic resampling.
    )


def validate_output(
    keras_model,
    keras_image_converter,
    keras_tokenizer,
    hf_model,
    hf_model_processor,
):
    """Validate that Keras Hub output matches HuggingFace output."""
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
    keras_preprocessor = keras_hub.models.MetaCLIP2CausalLMPreprocessor(
        keras_tokenizer
    )
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

    print(f"🏃 Converting {preset}")

    # Load HuggingFace model for validation.
    hf_model = AutoModel.from_pretrained(hf_preset)
    hf_preprocessor = AutoProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    # Load KerasHub model using on-the-fly HF conversion (via
    # keras_hub/src/utils/transformers/convert_metaclip_2.py).
    keras_model = keras_hub.models.MetaCLIP2Backbone.from_preset(
        f"hf://{hf_preset}"
    )
    keras_tokenizer = keras_hub.models.MetaCLIP2Tokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    keras_image_converter = convert_image_converter(
        hf_preprocessor.image_processor
    )
    keras_model.summary()
    print("✅ KerasHub model loaded.")
    print("✅ Weights converted.")

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
