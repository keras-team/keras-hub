"""Converts PaddleOCR's Differentiable Binarization weights to Keras.

Setup:
```shell
pip install paddlepaddle
```

Run as:
```shell
python convert_diffbin_checkpoints.py
```
"""

import re
import tarfile
import urllib

import keras
import numpy as np
import paddle

import keras_hub

MODEL_URI = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar"


def build_model():
    """Instantiate a DiffBin architecture based on ResNet50_vd."""
    image_encoder = keras_hub.models.ResNetBackbone(
        input_conv_filters=[32, 32, 64],
        input_conv_kernel_sizes=[3, 3, 3],
        stackwise_num_filters=[64, 128, 256, 512],
        stackwise_num_blocks=[3, 4, 6, 3],
        stackwise_num_strides=[1, 2, 2, 2],
        block_type="bottleneck_block_vd",
    )
    backbone = keras_hub.models.DiffBinBackbone(image_encoder=image_encoder)
    image_converter = keras_hub.layers.DiffBinImageConverter(
        scale=(1.0 / 255.0),
        variance=np.array([0.229, 0.224, 0.225]) ** 2,
        mean=np.array([0.485, 0.456, 0.406]),
        image_size=(640, 640),
        crop_to_aspect_ratio=False,
    )
    preprocessor = keras_hub.models.DiffBinPreprocessor(
        image_converter=image_converter,
    )
    model = keras_hub.models.DiffBinImageTextDetector(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    return model


def map_residual_layer_name(name: str):
    """Translate a Keras ResNet_vd layer name to a PaddleClas ResNet
    layer name prefix for a residual block."""

    match = re.match(
        r"^stack(?P<stack>\d)_block(?P<block>\d+)_(?P<conv>\d)_(?P<type>bn|conv)",
        name,
    )
    if match is None:
        return None
    if match["conv"] == "0":
        conv_name = "short"
    else:
        conv_name = f"conv{int(match['conv'])-1}"
    if match["type"] == "conv":
        return (
            f"backbone.bb_{match['stack']}_{match['block']}."
            f"{conv_name}._conv"
        )
    elif match["type"] == "bn":
        return (
            f"backbone.bb_{match['stack']}_{match['block']}."
            f"{conv_name}._batch_norm"
        )
    else:
        return None


def map_layer_name(name: str):
    """Translate a Keras layer name to a PaddleClas layer name prefix."""
    resnet_stem_mapping = {
        # ResNet stem layers
        "conv1_conv": "backbone.conv1_1._conv",
        "conv1_bn": "backbone.conv1_1._batch_norm",
        "conv2_conv": "backbone.conv1_2._conv",
        "conv2_bn": "backbone.conv1_2._batch_norm",
        "conv3_conv": "backbone.conv1_3._conv",
        "conv3_bn": "backbone.conv1_3._batch_norm",
    }
    diffbin_mapping = {
        # Differentiable Binarization layers
        "head_prob_conv0_bn": "head.binarize.conv_bn1",
        "head_prob_conv0_weights": "head.binarize.conv1",
        "head_prob_conv1_bn": "head.binarize.conv_bn2",
        "head_prob_conv1_weights": "head.binarize.conv2",
        "head_prob_conv2_weights": "head.binarize.conv3",
        "head_thresh_conv0_bn": "head.thresh.conv_bn1",
        "head_thresh_conv0_weights": "head.thresh.conv1",
        "head_thresh_conv1_bn": "head.thresh.conv_bn2",
        "head_thresh_conv1_weights": "head.thresh.conv2",
        "head_thresh_conv2_weights": "head.thresh.conv3",
        "neck_lateral_p2": "neck.in2_conv",
        "neck_lateral_p3": "neck.in3_conv",
        "neck_lateral_p4": "neck.in4_conv",
        "neck_lateral_p5": "neck.in5_conv",
        "neck_featuremap_p2": "neck.p2_conv",
        "neck_featuremap_p3": "neck.p3_conv",
        "neck_featuremap_p4": "neck.p4_conv",
        "neck_featuremap_p5": "neck.p5_conv",
    }
    return (
        resnet_stem_mapping.get(name)
        or diffbin_mapping.get(name)
        or map_residual_layer_name(name)
    )


def set_batchnorm(paddle_weights, paddle_name_prefix, target_layer):
    target_layer.set_weights(
        [
            paddle_weights.pop(f"{paddle_name_prefix}.weight"),
            paddle_weights.pop(f"{paddle_name_prefix}.bias"),
            paddle_weights.pop(f"{paddle_name_prefix}._mean"),
            paddle_weights.pop(f"{paddle_name_prefix}._variance"),
        ]
    )


def set_conv(paddle_weights, paddle_name_prefix, target_layer):
    if target_layer.use_bias:
        target_layer.set_weights(
            [
                np.transpose(
                    paddle_weights.pop(f"{paddle_name_prefix}.weight"),
                    (2, 3, 1, 0),
                ),
                paddle_weights.pop(f"{paddle_name_prefix}.bias"),
            ]
        )
    else:
        target_layer.set_weights(
            [
                np.transpose(
                    paddle_weights.pop(f"{paddle_name_prefix}.weight"),
                    (2, 3, 1, 0),
                )
            ]
        )


def main():
    # fetch and load the PaddleOCR weights
    urllib.request.urlretrieve(MODEL_URI, "paddle_diffbin.tar")
    with tarfile.open("paddle_diffbin.tar", "r") as tar:
        tar.extractall(filter="data")

    paddle_weights = paddle.load(
        "det_r50_vd_db_v2.0_train/best_accuracy.pdparams"
    )

    # initialize the KerasHub model
    model = build_model()
    model.summary()
    model.backbone.summary()

    # copy model weights
    for layer in model.backbone.layers:
        if len(layer.trainable_weights):
            paddle_prefix = map_layer_name(layer.name)
            if paddle_prefix is None:
                raise ValueError("Unexpected layer name encountered in model")
            if isinstance(
                layer, (keras.layers.Conv2D, keras.layers.Conv2DTranspose)
            ):
                set_conv(paddle_weights, paddle_prefix, layer)
            elif isinstance(layer, keras.layers.BatchNormalization):
                set_batchnorm(paddle_weights, paddle_prefix, layer)
            else:
                raise TypeError("Unexpected layer type encountered in model")

    # make sure there are no unused parameters
    assert len(paddle_weights) == 0

    # save the model
    model.save_to_preset("diffbin_resnet50vd")


if __name__ == "__main__":
    main()
