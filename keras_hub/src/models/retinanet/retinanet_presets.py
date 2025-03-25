"""RetinaNet model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "retinanet_resnet50_fpn_coco": {
        "metadata": {
            "description": (
                "RetinaNet model with ResNet50 backbone fine-tuned on COCO in "
                "800x800 resolution."
            ),
            "params": 34121239,
            "path": "retinanet",
        },
        "kaggle_handle": "kaggle://keras/retinanet/keras/retinanet_resnet50_fpn_coco/3",
    },
    "retinanet_resnet50_fpn_v2_coco": {
        "metadata": {
            "description": (
                "RetinaNet model with ResNet50 backbone fine-tuned on COCO in "
                "800x800 resolution with FPN features created from P5 level."
            ),
            "params": 31558592,
            "path": "retinanet",
        },
        "kaggle_handle": "kaggle://keras/retinanet/keras/retinanet_resnet50_fpn_v2_coco/2",
    },
}
