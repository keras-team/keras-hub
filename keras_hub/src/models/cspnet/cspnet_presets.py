"""CSPNet preset configurations."""

backbone_presets = {
    "csp_darknet_53_ra_imagenet": {
        "metadata": {
            "description": (
                "A CSP-DarkNet (Cross-Stage-Partial) image classification model"
                " pre-trained on the Randomly Augmented ImageNet 1k dataset at "
                "a 256x256 resolution."
            ),
            "params": 27642184,
            "path": "cspnet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_53_ra_imagenet/2",
    },
    "csp_resnext_50_ra_imagenet": {
        "metadata": {
            "description": (
                "A CSP-ResNeXt (Cross-Stage-Partial) image classification model"
                " pre-trained on the Randomly Augmented ImageNet 1k dataset at "
                "a 256x256 resolution."
            ),
            "params": 20569896,
            "path": "cspnet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_resnext_50_ra_imagenet/1",
    },
    "csp_resnet_50_ra_imagenet": {
        "metadata": {
            "description": (
                "A CSP-ResNet (Cross-Stage-Partial) image classification model"
                " pre-trained on the Randomly Augmented ImageNet 1k dataset at "
                "a 256x256 resolution."
            ),
            "params": 21616168,
            "path": "cspnet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_resnet_50_ra_imagenet/1",
    },
    "darknet_53_imagenet": {
        "metadata": {
            "description": (
                "A DarkNet image classification model pre-trained on the"
                "ImageNet 1k dataset at a 256x256 resolution."
            ),
            "params": 41609928,
            "path": "cspnet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/darknet_53_imagenet/1",
    },
}
