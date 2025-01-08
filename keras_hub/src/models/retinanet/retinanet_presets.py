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
        "kaggle_handle": "kaggle://keras/retinanet/keras/retinanet_resnet50_fpn_coco/2",
    }
}
