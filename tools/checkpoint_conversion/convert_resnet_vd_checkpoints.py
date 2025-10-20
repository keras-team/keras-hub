#!/usr/bin/env python3

"""Converts ResNet_vd models from PaddleClas.

Usage: python3 convert_resnet_vd_checkpoints.py

ResNet_vd model weights from PaddleClas listed in `configurations` below will
be downloaded, saved as Keras model files and the resulting models will be
verified for numerical agreement with PaddleClas.

Requirements:
pip3 install -q git+https://github.com/keras-team/keras-hub.git
pip3 install -q paddleclas paddlepaddle
"""

import os
import re
import tarfile
import urllib.request

import keras
import numpy as np
import paddle
import paddleclas
from paddleclas.deploy.python import preprocess as pc_preproc
from PIL import Image

import keras_hub

"""Architecture Specifications"""

configurations = {
    "ResNet18_vd": {
        "stackwise_num_blocks": [2, 2, 2, 2],
        "block_type": "basic_block_vd",
    },
    "ResNet34_vd": {
        "stackwise_num_blocks": [3, 4, 6, 3],
        "block_type": "basic_block_vd",
    },
    "ResNet50_vd": {
        "stackwise_num_blocks": [3, 4, 6, 3],
        "block_type": "bottleneck_block_vd",
    },
    "ResNet50_vd_ssld": {
        "stackwise_num_blocks": [3, 4, 6, 3],
        "block_type": "bottleneck_block_vd",
    },
    "ResNet50_vd_ssld_v2": {
        "stackwise_num_blocks": [3, 4, 6, 3],
        "block_type": "bottleneck_block_vd",
    },
    "Fix_ResNet50_vd_ssld_v2": {
        "stackwise_num_blocks": [3, 4, 6, 3],
        "block_type": "bottleneck_block_vd",
    },
    "ResNet101_vd": {
        "stackwise_num_blocks": [3, 4, 23, 3],
        "block_type": "bottleneck_block_vd",
    },
    "ResNet101_vd_ssld": {
        "stackwise_num_blocks": [3, 4, 23, 3],
        "block_type": "bottleneck_block_vd",
    },
    "ResNet152_vd": {
        "stackwise_num_blocks": [3, 8, 36, 3],
        "block_type": "bottleneck_block_vd",
    },
    "ResNet200_vd": {
        "stackwise_num_blocks": [3, 12, 48, 3],
        "block_type": "bottleneck_block_vd",
    },
}


"""Download Files"""

# Create the directory if it doesn't exist
os.makedirs("pretrained_models", exist_ok=True)
base_url = "https://paddle-imagenet-models-name.bj.bcebos.com/"

for arch in configurations.keys():
    tar_file = f"{arch}_pretrained.tar"
    download_url = f"{base_url}{tar_file}"
    file_path = os.path.join("pretrained_models", tar_file)

    # Download the tar file
    print(f"Downloading {tar_file}...")
    urllib.request.urlretrieve(download_url, file_path)

    # Extract the tar file
    print(f"Extracting {tar_file}...")
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(path="pretrained_models", filter="data")


"""Model Conversion"""


def convert_paddle_to_keras(paddle_weights: dict, keras_model: keras.Model):
    """Ports a paddle weights dictionary to a Keras model."""

    def map_residual_layer_name(name: str):
        """Translate a Keras ResNet_vd layer name to a PaddleClas ResNet
        layer name prefix for a residual block."""
        branch_mapping = {
            # this suffix addresses the specific conv layer within a block
            0: "1",
            1: "2a",
            2: "2b",
            3: "2c",
        }
        match = re.match(
            r"^stack(?P<stack>\d)_block(?P<block>\d+)_(?P<conv>\d)_(?P<type>bn|conv)",
            name,
        )
        assert match is not None

        # ResNet models have two different formats of layer name encodings
        # in PaddleClas. first try a mapping in the form
        # stack2_block3_1_conv -> res4b2_branch2a
        paddle_address = (
            f"{int(match['stack']) + 2}b{int(match['block'])}"
            f"_branch{branch_mapping[int(match['conv'])]}"
        )
        if match["type"] == "bn":
            paddle_name = f"bn{paddle_address}"
        elif match["type"] == "conv":
            paddle_name = f"res{paddle_address}"
        if any(name.startswith(paddle_name) for name in paddle_weights):
            return paddle_name

        # if that was not successful, try a mapping like
        # stack2_block3_1_conv -> res4c_branch2a
        paddle_address = (
            f"{int(match['stack']) + 2}{'abcdefghijkl'[int(match['block'])]}"
            f"_branch{branch_mapping[int(match['conv'])]}"
        )
        if match["type"] == "bn":
            paddle_name = f"bn{paddle_address}"
        elif match["type"] == "conv":
            paddle_name = f"res{paddle_address}"
        return paddle_name

    def map_layer_name(name: str):
        """Translate a Keras ResNet_vd layer name to a PaddleClas ResNet layer
        name prefix."""
        mapping = {
            # stem layers
            "conv1_conv": "conv1_1",
            "conv1_bn": "bnv1_1",
            "conv2_conv": "conv1_2",
            "conv2_bn": "bnv1_2",
            "conv3_conv": "conv1_3",
            "conv3_bn": "bnv1_3",
        }
        return mapping.get(name) or map_residual_layer_name(name)

    def set_batchnorm_layer(
        paddle_name_prefix: str, target_layer: keras.layers.Layer
    ):
        """Assign Keras BatchNorm layer weigths from Paddle weights."""
        target_layer.set_weights(
            [
                paddle_weights.pop(f"{paddle_name_prefix}_scale"),
                paddle_weights.pop(f"{paddle_name_prefix}_offset"),
                paddle_weights.pop(f"{paddle_name_prefix}_mean"),
                paddle_weights.pop(f"{paddle_name_prefix}_variance"),
            ]
        )

    def set_conv_layer(
        paddle_name_prefix: str, target_layer: keras.layers.Layer
    ):
        """Assign Keras Conv2D layer weights from Paddle weights."""
        if target_layer.use_bias:
            target_layer.set_weights(
                [
                    np.transpose(
                        paddle_weights.pop(f"{paddle_name_prefix}_weights"),
                        (2, 3, 1, 0),
                    ),
                    paddle_weights.pop(f"{paddle_name_prefix}_bias"),
                ]
            )
        else:
            target_layer.set_weights(
                [
                    np.transpose(
                        paddle_weights.pop(f"{paddle_name_prefix}_weights"),
                        (2, 3, 1, 0),
                    )
                ]
            )

    def set_dense_layer(
        paddle_name_prefix: str, target_layer: keras.layers.Layer
    ):
        """Assign Keras Dense layer weights from Paddle weights."""
        if target_layer.use_bias:
            target_layer.set_weights(
                [
                    paddle_weights.pop(f"{paddle_name_prefix}.w_0"),
                    paddle_weights.pop(f"{paddle_name_prefix}.b_0"),
                ]
            )
        else:
            target_layer.set_weights(
                [paddle_weights.pop(f"{paddle_name_prefix}.w_0")]
            )

    for layer in keras_model.backbone.layers:
        # iterate over all layers that have parameters in the keras model,
        # to ensure we process all weights in the Keras model
        if layer.variables:
            if isinstance(layer, keras.layers.Conv2D):
                set_conv_layer(map_layer_name(layer.name), layer)
            elif isinstance(layer, keras.layers.BatchNormalization):
                set_batchnorm_layer(map_layer_name(layer.name), layer)
            else:
                raise TypeError("Unexpected layer type encountered in model")
    set_dense_layer("fc_0", keras_model.get_layer("predictions"))

    # ensure we have consumed all weights, i.e. there are no leftover
    # weights in the paddle model
    assert len(paddle_weights) == 0


"""Instantiate model architectures as indicated above and load PaddleClas
weights into the Keras model"""

for architecture_name, architecture_config in configurations.items():
    print(f"Converting {architecture_name}")
    backbone_model = keras_hub.models.ResNetBackbone(
        input_conv_filters=[32, 32, 64],
        input_conv_kernel_sizes=[3, 3, 3],
        stackwise_num_filters=[64, 128, 256, 512],
        stackwise_num_strides=[1, 2, 2, 2],
        **architecture_config,
    )
    image_converter = keras_hub.layers.ResNetImageConverter(
        height=224,
        width=224,
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],
        scale=1 / 255.0,
    )
    resnet_preprocessor = keras_hub.models.ResNetImageClassifierPreprocessor(
        image_converter
    )
    classifier_model = keras_hub.models.ResNetImageClassifier(
        backbone=backbone_model,
        preprocessor=resnet_preprocessor,
        num_classes=1000,
    )
    paddle_model = paddle.load(
        f"pretrained_models/{architecture_name}_pretrained"
    )
    convert_paddle_to_keras(paddle_model, classifier_model)
    classifier_model.save(f"{architecture_name}.keras")
    classifier_model.save_to_preset(f"{architecture_name}")
    print(f"Parameter count: {classifier_model.count_params()}")

"""Check for Numerical Agreement

Compare results when using PaddleClas with results when using our Keras models.
In general, PaddleClas appears to mainly target command-line utilisation
rather than offering an API. While PaddleClas model architectures can directly
be instantiated, this interface strangely only provides some of the pretrained
models (and doesn't appear to be documented anywhere).

To ensure behaviour and performances when using PaddleClas as command-line tool
match our observed results, we here use `PaddleClas` directly.
"""

urllib.request.urlretrieve(
    "https://storage.googleapis.com/tensorflow/keras-applications/tests/elephant.jpg",
    "elephant.jpg",
)

print(f"{'Model': <25}Error")
for architecture_name in configurations:
    # PaddleClas prediction
    predictor = paddleclas.PaddleClas(model_name=architecture_name).predictor
    # PaddleClas selects the top 5 predictions during
    # postprocessing. turn this off.
    predictor.postprocess = None
    # for comparable results, manually perform resizing and cropping
    preprocess_ops = [
        op
        for op in predictor.preprocess_ops
        if isinstance(
            op,
            (
                pc_preproc.NormalizeImage,
                pc_preproc.ResizeImage,
                pc_preproc.CropImage,
            ),
        )
    ]
    predictor.preprocess_ops = [
        op for op in predictor.preprocess_ops if op not in preprocess_ops
    ]
    image = np.asarray(Image.open("elephant.jpg"), dtype=np.float32)
    for op in preprocess_ops:
        image = op(image)
    paddle_prediction = predictor.predict(image)

    # Keras prediction
    # in contrast to PaddleClas, Keras' predictions are not softmax'ed
    keras_model = keras.saving.load_model(f"{architecture_name}.keras")
    keras_prediction = keras_model(image[None]).numpy()
    keras_prediction = keras.ops.softmax(keras_prediction)

    # compare
    max_error = np.max(np.abs(paddle_prediction - keras_prediction))
    print(f"{architecture_name: <25}{max_error}")
