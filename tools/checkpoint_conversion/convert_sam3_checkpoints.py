"""Convert SigLIP checkpoints.

export KAGGLE_USERNAME=xxx
export KAGGLE_KEY=xxx

python tools/checkpoint_conversion/convert_sam3_checkpoints.py \
    --preset sam3_pcs --upload_uri kaggle://kerashub/sam3/keras/sam3_pcs
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from PIL import Image
from transformers.models.sam3 import Sam3Model
from transformers.models.sam3 import Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3ImageSegmentationOutput

import keras_hub
from keras_hub.src.models.sam3.sam3_image_converter import SAM3ImageConverter
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)
from keras_hub.src.models.sam3.sam3_pc_image_segmenter_preprocessor import (
    SAM3PromptableConceptImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam3.sam3_tokenizer import SAM3Tokenizer

FLAGS = flags.FLAGS

PRESET_MAP = {
    "sam3_pcs": "facebook/sam3",
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
    config = hf_image_processor.to_dict()
    image_size = (config["size"]["height"], config["size"]["width"])
    std = config["image_std"]
    mean = config["image_mean"]
    return SAM3ImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        crop_to_aspect_ratio=False,
        antialias=True,
    )


def convert_tokenizer(hf_tokenizer):
    vocabulary = hf_tokenizer.vocab
    merges = [" ".join(item) for item in hf_tokenizer._merges]
    return SAM3Tokenizer(vocabulary, merges, name="sam3_tokenizer")


def validate_output(
    keras_hub_model,
    keras_hub_preprocessor,
    hf_model,
    hf_preprocessor,
):
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000077595.jpg")
    )
    image = Image.open(file).convert("RGB")
    input_boxes = [[[0, 0, 300, 430]]]
    input_boxes_labels = [[1]]

    # Preprocess with hf.
    hf_inputs = hf_preprocessor(
        images=image,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_tensors="pt",
    )

    # Preprocess with keras.
    keras_inputs = keras_hub_preprocessor(
        {
            "images": np.expand_dims(np.array(image), axis=0),
            "boxes": input_boxes,
            "box_labels": input_boxes_labels,
        }
    )

    keras_token_ids = keras.ops.convert_to_numpy(keras_inputs["token_ids"])
    hf_token_ids = hf_inputs["input_ids"].cpu().numpy()
    keras_pixel_values = keras.ops.convert_to_numpy(
        keras_inputs["pixel_values"]
    )
    hf_pixel_values = hf_inputs["pixel_values"].cpu().numpy()
    keras_boxes = keras.ops.convert_to_numpy(keras_inputs["boxes"])
    hf_boxes = hf_inputs["input_boxes"].cpu().numpy()
    print(f"🔶 Keras preprocessor token_ids: {keras_token_ids[0, :5]} ...")
    print(f"🔶 HF preprocessor token_ids.  : {hf_token_ids[0, :5]} ...")
    print(
        "🔶 Keras preprocessor pixel_values:\n "
        f"{keras_pixel_values[0, :3, :3, 0]}"
    )
    print(f"🔶 HF preprocessor pixel_values:\n {hf_pixel_values[0, 0, :3, :3]}")
    print(f"🔶 Keras preprocessor boxes: {keras_boxes[0, :, 1:]}")
    print(f"🔶 HF preprocessor boxes.  : {hf_boxes[0]}")

    # Call with hf. Use the keras preprocessed image so we can keep modeling
    # and preprocessing comparisons independent.
    hf_inputs["pixel_values"] = torch.from_numpy(
        keras.ops.convert_to_numpy(
            keras.ops.transpose(keras_pixel_values, (0, 3, 1, 2))
        )
    )
    with torch.inference_mode():
        hf_outputs = hf_model(**hf_inputs)
    hf_outputs = hf_preprocessor.post_process_instance_segmentation(
        hf_outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=hf_inputs.get("original_sizes").tolist(),
    )[0]
    hf_output_scores = hf_outputs["scores"].cpu().numpy()
    hf_output_boxes = hf_outputs["boxes"].cpu().numpy()

    # Call with keras. Using HF processor to post-process outputs.
    keras_outputs = keras_hub_model.predict(keras_inputs, verbose=0)
    keras_outputs = hf_preprocessor.post_process_instance_segmentation(
        Sam3ImageSegmentationOutput(
            pred_masks=torch.from_numpy(
                keras.ops.convert_to_numpy(
                    keras_outputs["pred_masks"]
                ).transpose(0, 3, 1, 2)
            ),
            pred_boxes=torch.from_numpy(
                keras.ops.convert_to_numpy(keras_outputs["pred_boxes"])
            ),
            pred_logits=torch.from_numpy(
                keras.ops.convert_to_numpy(keras_outputs["pred_logits"])
            ),
            presence_logits=torch.from_numpy(
                keras.ops.convert_to_numpy(keras_outputs["presence_logits"])
            ),
        ),
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=hf_inputs.get("original_sizes").tolist(),
    )[0]
    keras_output_scores = keras.ops.convert_to_numpy(keras_outputs["scores"])
    keras_output_boxes = keras.ops.convert_to_numpy(keras_outputs["boxes"])

    print(f"🔶 Keras output scores: {keras_output_scores}")
    print(f"🔶 HF output scores.  : {hf_output_scores}")
    print(
        f"🔶 Keras output boxes: "
        f"{np.array_str(keras_output_boxes, precision=3, suppress_small=True)}"
    )
    print(
        f"🔶 HF output boxes.  : "
        f"{np.array_str(hf_output_boxes, precision=3, suppress_small=True)}"
    )
    boxes_diff = np.mean(np.abs(keras_output_boxes - hf_output_boxes))
    print(f"🔶 Modeling box difference: {boxes_diff}")
    preprocessing_diff = np.mean(
        np.abs(keras_pixel_values - np.transpose(hf_pixel_values, (0, 2, 3, 1)))
    )
    print(f"🔶 Preprocessing difference: {preprocessing_diff}")


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
    hf_model = Sam3Model.from_pretrained(hf_preset)
    hf_preprocessor = Sam3Processor.from_pretrained(hf_preset)
    hf_model.eval()

    if "pcs" in preset:
        keras_hub_backbone: SAM3PromptableConceptBackbone = (
            SAM3PromptableConceptBackbone.from_preset(f"hf://{hf_preset}")
        )
    else:
        # TODO: Add PVS.
        raise ValueError(f"Unsupported preset {hf_preset}")
    keras_hub_backbone.summary()
    keras_hub_image_converter = convert_image_converter(
        hf_preprocessor.image_processor
    )
    keras_hub_tokenizer = convert_tokenizer(hf_preprocessor.tokenizer)
    keras_hub_preprocessor = SAM3PromptableConceptImageSegmenterPreprocessor(
        keras_hub_tokenizer, keras_hub_image_converter
    )
    print("✅ KerasHub model loaded.")
    print("✅ Weights converted.")

    validate_output(
        keras_hub_backbone,
        keras_hub_preprocessor,
        hf_model,
        hf_preprocessor,
    )
    print("✅ Output validated.")

    keras_hub_backbone.save_to_preset(f"./{preset}")
    keras_hub_image_converter.save_to_preset(f"./{preset}")
    keras_hub_tokenizer.save_to_preset(f"./{preset}")
    keras_hub_preprocessor.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
