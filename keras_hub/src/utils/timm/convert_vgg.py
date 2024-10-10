from typing import Any

import numpy as np

from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone
from keras_hub.src.models.vgg.vgg_image_classifier import VGGImageClassifier

backbone_cls = VGGBackbone


REPEATS_BY_SIZE = {
    "vgg11": [1, 1, 2, 2, 2],
    "vgg13": [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}


def convert_backbone_config(timm_config):
    architecture = timm_config["architecture"]
    stackwise_num_repeats = REPEATS_BY_SIZE[architecture]
    return dict(
        stackwise_num_repeats=stackwise_num_repeats,
        stackwise_num_filters=[64, 128, 256, 512, 512],
    )


def convert_conv2d(
    model,
    loader,
    keras_layer_name: str,
    hf_layer_name: str,
):
    loader.port_weight(
        model.get_layer(keras_layer_name).kernel,
        hf_weight_key=f"{hf_layer_name}.weight",
        hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
    )
    loader.port_weight(
        model.get_layer(keras_layer_name).bias,
        hf_weight_key=f"{hf_layer_name}.bias",
    )


def convert_weights(
    backbone: VGGBackbone,
    loader,
    timm_config: dict[Any],
):
    architecture = timm_config["architecture"]
    stackwise_num_repeats = REPEATS_BY_SIZE[architecture]

    hf_index_to_keras_layer_name = {}
    layer_index = 0
    for block_index, repeats_in_block in enumerate(stackwise_num_repeats):
        for repeat_index in range(repeats_in_block):
            hf_index = layer_index
            layer_index += 2  # Conv + activation layers.
            layer_name = f"block{block_index + 1}_conv{repeat_index + 1}"
            hf_index_to_keras_layer_name[hf_index] = layer_name
        layer_index += 1  # Pooling layer after blocks.

    for hf_index, keras_layer_name in hf_index_to_keras_layer_name.items():
        convert_conv2d(
            backbone, loader, keras_layer_name, f"features.{hf_index}"
        )


def convert_head(
    task: VGGImageClassifier,
    loader,
    timm_config: dict[Any],
):
    convert_conv2d(task.head, loader, "fc1", "pre_logits.fc1")
    convert_conv2d(task.head, loader, "fc2", "pre_logits.fc2")

    loader.port_weight(
        task.head.get_layer("predictions").kernel,
        hf_weight_key="head.fc.weight",
        hook_fn=lambda x, _: np.transpose(np.squeeze(x)),
    )
    loader.port_weight(
        task.head.get_layer("predictions").bias,
        hf_weight_key="head.fc.bias",
    )
