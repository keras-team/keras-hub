"""CSPNet preset configurations."""

backbone_presets = {
    "csp_darknet_53_ra_imagenet": {
        "metadata": {
            "description": (
                "A CSP-DarkNet (Cross-Stage-Partial) image classification model"
                " pre-trained on the Randomly Augmented ImageNet 1k dataset at "
                "a 224x224 resolution."
            ),
            "params": 26652512,
            "path": "cspnet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_53_ra_imagenet/1",
    },
}
