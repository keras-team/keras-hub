"""Convert DepthAnything checkpoints.

export KAGGLE_USERNAME=xxx
export KAGGLE_KEY=xxx

python tools/checkpoint_conversion/convert_depthanything_checkpoints.py \
    --preset depth_anything_v2_small --upload_uri kaggle://kerashub/depth_anything/keras/depth_anything_v2_small
python tools/checkpoint_conversion/convert_depthanything_checkpoints.py \
    --preset depth_anything_v2_base --upload_uri kaggle://kerashub/depth_anything/keras/depth_anything_v2_base
python tools/checkpoint_conversion/convert_depthanything_checkpoints.py \
    --preset depth_anything_v2_large --upload_uri kaggle://kerashub/depth_anything/keras/depth_anything_v2_large
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
from transformers import DepthAnythingForDepthEstimation

import keras_hub
from keras_hub.src.models.depth_anything.depth_anything_backbone import (
    DepthAnythingBackbone,
)
from keras_hub.src.models.depth_anything.depth_anything_image_converter import (
    DepthAnythingImageConverter,
)
from keras_hub.src.models.dinov2.dinov2_backbone import DINOV2Backbone
from keras_hub.src.utils.transformers import convert_dinov2
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

FLAGS = flags.FLAGS

PRESET_MAP = {
    "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
    "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
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


def convert_model(hf_model, dtype=None):
    dinov2_config = convert_dinov2.convert_backbone_config(
        hf_model.config.backbone_config.to_dict()
    )
    image_encoder = DINOV2Backbone(**dinov2_config)
    model_config = hf_model.config.to_dict()
    # In KerasHub, the stage names are capitalized.
    feature_keys = model_config["backbone_config"]["out_features"]
    feature_keys = [key.replace("stage", "Stage") for key in feature_keys]
    assert model_config["depth_estimation_type"] == "relative"
    assert model_config["max_depth"] in (None, 1.0)
    return DepthAnythingBackbone(
        image_encoder,
        reassemble_factors=model_config["reassemble_factors"],
        neck_hidden_dims=model_config["neck_hidden_sizes"],
        fusion_hidden_dim=model_config["fusion_hidden_size"],
        head_hidden_dim=model_config["head_hidden_size"],
        head_in_index=model_config["head_in_index"],
        feature_keys=feature_keys,
        dtype=dtype,
    )


def convert_weights(hf_preset, keras_hub_model, hf_model):
    # Convert weights of DINOV2 backbone.
    with SafetensorLoader(f"hf://{hf_preset}") as loader:
        convert_dinov2.convert_weights(
            keras_hub_model.image_encoder, loader, None
        )

    # Get `state_dict` from `hf_model`.
    state_dict = hf_model.state_dict()

    # Helper functions.
    def port_weights(keras_variable, weight_key, hook_fn=None):
        torch_tensor = state_dict[weight_key].cpu().numpy()
        if hook_fn:
            torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
        keras_variable.assign(torch_tensor)

    def port_conv2d(keras_variable, weight_key):
        port_weights(
            keras_variable.kernel,
            f"{weight_key}.weight",
            lambda x, s: np.transpose(x, (2, 3, 1, 0)),
        )
        if keras_variable.use_bias:
            port_weights(keras_variable.bias, f"{weight_key}.bias")

    assert isinstance(keras_hub_model, DepthAnythingBackbone)

    # Convert neck weights.
    for i in range(len(keras_hub_model.reassemble_factors)):
        # Reassemble stage.
        port_conv2d(
            keras_hub_model.neck.reassemble_stage[i].projection,
            f"neck.reassemble_stage.layers.{i}.projection",
        )
        if keras_hub_model.neck.reassemble_stage[i].factor != 1:
            port_conv2d(
                keras_hub_model.neck.reassemble_stage[i].resize,
                f"neck.reassemble_stage.layers.{i}.resize",
            )
        # Convs.
        port_conv2d(keras_hub_model.neck.convs[i], f"neck.convs.{i}")
        # Fusion stage.
        port_conv2d(
            keras_hub_model.neck.fusion_stage[i].projection,
            f"neck.fusion_stage.layers.{i}.projection",
        )
        port_conv2d(
            keras_hub_model.neck.fusion_stage[i].residual_layer1.convolution1,
            f"neck.fusion_stage.layers.{i}.residual_layer1.convolution1",
        )
        port_conv2d(
            keras_hub_model.neck.fusion_stage[i].residual_layer1.convolution2,
            f"neck.fusion_stage.layers.{i}.residual_layer1.convolution2",
        )
        port_conv2d(
            keras_hub_model.neck.fusion_stage[i].residual_layer2.convolution1,
            f"neck.fusion_stage.layers.{i}.residual_layer2.convolution1",
        )
        port_conv2d(
            keras_hub_model.neck.fusion_stage[i].residual_layer2.convolution2,
            f"neck.fusion_stage.layers.{i}.residual_layer2.convolution2",
        )

    # Convert head weights.
    port_conv2d(keras_hub_model.head.conv1, "head.conv1")
    port_conv2d(keras_hub_model.head.conv2, "head.conv2")
    port_conv2d(keras_hub_model.head.conv3, "head.conv3")


def convert_image_converter(hf_image_processor):
    config = hf_image_processor.to_dict()
    image_size = (config["size"]["height"], config["size"]["width"])
    std = config["image_std"]
    mean = config["image_mean"]
    return DepthAnythingImageConverter(
        image_size=image_size,
        scale=[1.0 / 255.0 / s for s in std],
        offset=[-m / s for m, s in zip(mean, std)],
        interpolation="bicubic",  # DINOV2 defaults to bicubic resampling.
    )


def validate_output(
    keras_model, keras_image_converter, hf_model, hf_image_processor
):
    config = hf_image_processor.to_dict()
    image_size = (config["size"]["height"], config["size"]["width"])
    file = keras.utils.get_file(
        origin=("http://images.cocodataset.org/val2017/000000039769.jpg")
    )
    image = Image.open(file)
    image = image.resize(image_size)

    # Preprocess with hf.
    hf_inputs = hf_image_processor(images=image, return_tensors="pt")
    hf_preprocessed = hf_inputs["pixel_values"].detach().cpu().numpy()

    # Preprocess with keras.
    images = np.expand_dims(np.array(image).astype("float32"), axis=0)
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
    hf_depths = hf_outputs.predicted_depth.detach().cpu().numpy()

    # Call with keras.
    keras_depths = keras_model.predict(images, verbose=0)
    # Defaults to "relative" depth estimation.
    keras_depths = keras.ops.nn.relu(keras_depths)
    keras_depths = keras.ops.convert_to_numpy(
        keras.ops.squeeze(keras_depths, axis=-1)
    )

    print("🔶 Keras output:", keras_depths[0])
    print("🔶 HF output:", hf_depths[0])
    modeling_diff = np.mean(np.abs(keras_depths - hf_depths))
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
    hf_model = DepthAnythingForDepthEstimation.from_pretrained(hf_preset)
    hf_image_converter = AutoImageProcessor.from_pretrained(hf_preset)
    hf_model.eval()

    keras_model = convert_model(hf_model)
    keras_model.summary()
    keras_image_converter = convert_image_converter(hf_image_converter)
    print("✅ KerasHub model loaded.")

    convert_weights(hf_preset, keras_model, hf_model)
    print("✅ Weights converted.")

    validate_output(
        keras_model,
        keras_image_converter,
        hf_model,
        hf_image_converter,
    )
    print("✅ Output validated.")

    keras_model.save_to_preset(f"./{preset}")
    keras_image_converter.save_to_preset(f"./{preset}")
    print(f"🏁 Preset saved to ./{preset}.")

    upload_uri = FLAGS.upload_uri
    if upload_uri:
        keras_hub.upload_preset(uri=upload_uri, preset=f"./{preset}")
        print(f"🏁 Preset uploaded to {upload_uri}")


if __name__ == "__main__":
    app.run(main)
