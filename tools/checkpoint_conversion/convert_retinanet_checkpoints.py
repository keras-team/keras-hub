"""Convert ViT checkpoints.

export KAGGLE_USERNAME=XXX
export KAGGLE_KEY=XXX

python tools/checkpoint_conversion/convert_retinanet_checkpoints.py \
    --preset retinanet_resnet50_fpn_coco
"""

import os
import shutil

import keras
import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.retinanet import (
    RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.models.retinanet.retinanet_image_converter import (
    RetinaNetImageConverter,
)
from keras_hub.src.models.retinanet.retinanet_object_detector import (
    RetinaNetObjectDetector,
)
from keras_hub.src.models.retinanet.retinanet_object_detector_preprocessor import (  # noqa: E501
    RetinaNetObjectDetectorPreprocessor,
)

FLAGS = flags.FLAGS

PRESET_MAP = {
    "retinanet_resnet50_fpn_coco": RetinaNet_ResNet50_FPN_Weights.DEFAULT,
    "retinanet_resnet50_fpn_v2_coco": RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
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


def get_keras_backbone(use_p5=False):
    image_encoder = Backbone.from_preset(
        "resnet_50_imagenet", load_weights=False
    )
    backbone = RetinaNetBackbone(
        image_encoder=image_encoder,
        min_level=3,
        max_level=7,
        use_p5=use_p5,
    )

    return backbone


# Helper functions.
def port_weight(keras_variable, torch_tensor, hook_fn=None):
    if hook_fn:
        torch_tensor = hook_fn(torch_tensor, list(keras_variable.shape))
    keras_variable.assign(torch_tensor)


def convert_image_encoder(state_dict, backbone):
    def port_conv2d(keras_layer_name, torch_weight_prefix):
        port_weight(
            backbone.get_layer(keras_layer_name).kernel,
            torch_tensor=state_dict[f"{torch_weight_prefix}.weight"],
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )

    def port_batch_normalization(keras_layer_name, torch_weight_prefix):
        port_weight(
            backbone.get_layer(keras_layer_name).gamma,
            torch_tensor=state_dict[f"{torch_weight_prefix}.weight"],
        )
        port_weight(
            backbone.get_layer(keras_layer_name).beta,
            torch_tensor=state_dict[f"{torch_weight_prefix}.bias"],
        )
        port_weight(
            backbone.get_layer(keras_layer_name).moving_mean,
            torch_tensor=state_dict[f"{torch_weight_prefix}.running_mean"],
        )
        port_weight(
            backbone.get_layer(keras_layer_name).moving_variance,
            torch_tensor=state_dict[f"{torch_weight_prefix}.running_var"],
        )

    block_type = backbone.block_type

    # Stem
    port_conv2d("conv1_conv", "backbone.body.conv1")
    port_batch_normalization("conv1_bn", "backbone.body.bn1")

    # Stages
    num_stacks = len(backbone.stackwise_num_filters)
    for stack_index in range(num_stacks):
        for block_idx in range(backbone.stackwise_num_blocks[stack_index]):
            keras_name = f"stack{stack_index}_block{block_idx}"
            torch_name = f"backbone.body.layer{stack_index + 1}.{block_idx}"

            if block_idx == 0 and (
                block_type == "bottleneck_block" or stack_index > 0
            ):
                port_conv2d(
                    f"{keras_name}_0_conv", f"{torch_name}.downsample.0"
                )
                port_batch_normalization(
                    f"{keras_name}_0_bn", f"{torch_name}.downsample.1"
                )
            port_conv2d(f"{keras_name}_1_conv", f"{torch_name}.conv1")
            port_batch_normalization(f"{keras_name}_1_bn", f"{torch_name}.bn1")
            port_conv2d(f"{keras_name}_2_conv", f"{torch_name}.conv2")
            port_batch_normalization(f"{keras_name}_2_bn", f"{torch_name}.bn2")
            if block_type == "bottleneck_block":
                port_conv2d(f"{keras_name}_3_conv", f"{torch_name}.conv3")
                port_batch_normalization(
                    f"{keras_name}_3_bn", f"{torch_name}.bn3"
                )


def convert_fpn(state_dict, fpn_network):
    def port_conv2d(kera_weight, torch_weight_prefix):
        port_weight(
            kera_weight.kernel,
            torch_tensor=state_dict[f"{torch_weight_prefix}.weight"],
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        port_weight(
            kera_weight.bias,
            torch_tensor=state_dict[f"{torch_weight_prefix}.bias"],
        )

    for level, layer in fpn_network.lateral_conv_layers.items():
        idx = int(level[1])
        port_conv2d(layer, f"backbone.fpn.inner_blocks.{idx - 3}.0")

    for level, layer in fpn_network.output_conv_layers.items():
        idx = int(level[1])
        if "output" in layer.name:
            port_conv2d(layer, f"backbone.fpn.layer_blocks.{idx - 3}.0")
        if "coarser" in layer.name:
            port_conv2d(layer, f"backbone.fpn.extra_blocks.p{idx}")


def convert_head_weights(state_dict, keras_model):
    def port_conv2d(kera_weight, torch_weight_prefix):
        port_weight(
            kera_weight.kernel,
            torch_tensor=state_dict[f"{torch_weight_prefix}.weight"],
            hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
        )
        
        port_weight(
            kera_weight.bias,
            torch_tensor=state_dict[f"{torch_weight_prefix}.bias"],
        )

    for idx, layer in enumerate(keras_model.box_head.conv_layers):
        if FLAGS.preset == "retinanet_resnet50_fpn_coco":
            port_conv2d(layer, f"head.regression_head.conv.{idx}.0")
        else:
            port_weight(
              layer.kernel,
              torch_tensor=state_dict[f"head.regression_head.conv.{idx}.0.weight"],
              hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
            )

    for idx, layer in enumerate(keras_model.box_head.group_norm_layers):
        port_weight(
            layer.gamma,
            state_dict[f"head.regression_head.conv.{idx}.1.weight"],
        )
        port_weight(
            layer.beta, state_dict[f"head.regression_head.conv.{idx}.1.bias"]
        )

    port_conv2d(
        keras_model.box_head.prediction_layer,
        torch_weight_prefix="head.regression_head.bbox_reg",
    )
    for idx, layer in enumerate(keras_model.classification_head.conv_layers):
        if FLAGS.preset == "retinanet_resnet50_fpn_coco":
            port_conv2d(layer, f"head.classification_head.conv.{idx}.0")
        else:
            port_weight(
                layer.kernel,
                torch_tensor=state_dict[f"head.classification_head.conv.{idx}.0.weight"],
                hook_fn=lambda x, _: np.transpose(x, (2, 3, 1, 0)),
            )

    for idx, layer in enumerate(
        keras_model.classification_head.group_norm_layers
    ):
        port_weight(
            layer.gamma,
            state_dict[f"head.classification_head.conv.{idx}.1.weight"],
        )
        port_weight(
            layer.beta,
            state_dict[f"head.classification_head.conv.{idx}.1.bias"],
        )

    port_conv2d(
        keras_model.classification_head.prediction_layer,
        torch_weight_prefix="head.classification_head.cls_logits",
    )


def convert_backbone_weights(state_dict, backbone):
    # Convert ResNet50 image encoder
    convert_image_encoder(state_dict, backbone.image_encoder)
    # Convert FPN
    convert_fpn(state_dict, backbone.feature_pyramid)


def convert_image_converter(torch_model):
    image_mean = torch_model.transform.image_mean
    image_std = torch_model.transform.image_std
    resolution = torch_model.transform.min_size[0]
    return RetinaNetImageConverter(
        image_size=(resolution, resolution),
        pad_to_aspect_ratio=True,
        crop_to_aspect_ratio=False,
        scale=[1.0 / 255.0 / s for s in image_std],
        offset=[-m / s for m, s in zip(image_mean, image_std)],
    )


def main(_):
    if FLAGS.preset not in PRESET_MAP.keys():
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one "
            f"of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    torch_preset = PRESET_MAP[preset]
    if os.path.exists(preset):
        shutil.rmtree(preset)
    os.makedirs(preset)

    print(f"🏃 Coverting {preset}")

    # Load huggingface model.
    if preset == "retinanet_resnet50_fpn_coco":
        torch_model = retinanet_resnet50_fpn(weights=torch_preset)
        torch_model.eval()
        keras_backbone = get_keras_backbone()
    elif preset == "retinanet_resnet50_fpn_v2_coco":
        torch_model = retinanet_resnet50_fpn_v2(weights=torch_preset)
        torch_model.eval()
        keras_backbone = get_keras_backbone(use_p5=True)

    state_dict = torch_model.state_dict()
    print("✅ Torch and KerasHub model loaded.")

    convert_backbone_weights(state_dict, keras_backbone)
    print("✅ Backbone weights converted.")

    keras_image_converter = convert_image_converter(torch_model)
    print("✅ Loaded image converter")

    preprocessor = RetinaNetObjectDetectorPreprocessor(
        image_converter=keras_image_converter
    )

    keras_model = RetinaNetObjectDetector(
        backbone=keras_backbone,
        num_classes=len(torch_preset.meta["categories"]),
        preprocessor=preprocessor,
        use_prediction_head_norm=True
        if preset == "retinanet_resnet50_fpn_v2_coco"
        else False,
    )

    convert_head_weights(state_dict, keras_model)
    print("✅ Loaded head weights")

    filepath = keras.utils.get_file(
        origin="http://farm4.staticflickr.com/3755/10245052896_958cbf4766_z.jpg"
    )
    image = keras.utils.load_img(filepath)
    image = ops.cast(image, "float32")
    image = ops.expand_dims(image, axis=0)
    keras_image = preprocessor(image)
    torch_image = ops.transpose(keras_image, axes=(0, 3, 1, 2))
    torch_image = ops.convert_to_numpy(torch_image)
    torch_image = torch.from_numpy(torch_image)

    keras_outputs = keras_model(keras_image)
    with torch.no_grad():
        torch_mid_outputs = list(torch_model.backbone(torch_image).values())
        torch_outputs = torch_model.head(torch_mid_outputs)

    bbox_diff = np.mean(
        np.abs(
            ops.convert_to_numpy(keras_outputs["bbox_regression"])
            - torch_outputs["bbox_regression"].numpy()
        )
    )
    cls_logits_diff = np.mean(
        np.abs(
            ops.convert_to_numpy(keras_outputs["cls_logits"])
            - torch_outputs["cls_logits"].numpy()
        )
    )
    print("🔶 Modeling Bounding Box Logits difference:", bbox_diff)
    print("🔶 Modeling Class Logits difference:", cls_logits_diff)


if __name__ == "__main__":
    app.run(main)
