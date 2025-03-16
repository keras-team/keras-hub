"""InceptionNet preset configurations."""

backbone_presets = {
    "inception_v1_imagenet": {
        "metadata": {
            "description": (
                "InceptionV1 (GoogLeNet) model pre-trained on the"
                "ImageNet 1k dataset "
                "at a 224x224 resolution."
            ),
            "params": 6998552,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inceptionv1/keras/inception_v1_imagenet/1",
    },
    "inception_v2_imagenet": {
        "metadata": {
            "description": (
                "InceptionV2 model pre-trained on the ImageNet 1k dataset "
                "at a 224x224 resolution. Includes batch normalization."
            ),
            "params": 11268392,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inceptionv2/keras/inception_v2_imagenet/1",
    },
    "inception_v3_imagenet": {
        "metadata": {
            "description": (
                "InceptionV3 model pre-trained on the ImageNet 1k dataset "
                "at a 299x299 resolution. Features factorized convolutions and"
                "improved pooling strategies."
            ),
            "params": 23851784,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inceptionv3/keras/inception_v3_imagenet/1",
    },
    "inception_v4_imagenet": {
        "metadata": {
            "description": (
                "InceptionV4 model pre-trained on the ImageNet 1k dataset "
                "at a 299x299 resolution. Features a more uniform architecture "
                "with more inception modules."
            ),
            "params": 42679816,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inceptionv4/keras/inception_v4_imagenet/1",
    },
    "inception_resnet_v2_imagenet": {
        "metadata": {
            "description": (
                "Inception-ResNet-v2 hybrid model pre-trained on the"
                "Imagenet 1k dataset at a 299x299 resolution.Combines Inception"
                "architecture with residual connections."
            ),
            "params": 55873736,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inception_resnet/keras/inception_resnet_v2_imagenet/1",
    },
    "inception_v3_transfer_imagenet": {
        "metadata": {
            "description": (
                "InceptionV3 model pre-trained on the ImageNet 1k dataset "
                "at a 299x299 resolution, fine-tuned with transfer learning"
                "techniques for improved accuracy."
            ),
            "params": 23851784,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inceptionv3/keras/inception_v3_transfer_imagenet/1",
    },
    "inception_v3_augmented_imagenet": {
        "metadata": {
            "description": (
                "InceptionV3 model pre-trained on the ImageNet 1k dataset "
                "at a 299x299 resolution with extensive data augmentation for "
                "improved generalization."
            ),
            "params": 23851784,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inceptionv3/keras/inception_v3_augmented_imagenet/1",
    },
    "inception_v4_transfer_imagenet": {
        "metadata": {
            "description": (
                "InceptionV4 model pre-trained on the ImageNet 1k dataset "
                "at a 299x299 resolution, fine-tuned with transfer learning"
                "techniques for improved accuracy."
            ),
            "params": 42679816,
            "path": "inception",
        },
        "kaggle_handle": 
        "kaggle://keras/inceptionv4/keras/inception_v4_transfer_imagenet/1",
    },
    "inception_resnet_v2_transfer_imagenet": {
     "metadata": {
            "description": (
             "Inception-ResNet-v2 hybrid model pre-trained on the ImageNet 1k"
             "dataset at a 299x299 resolution with transfer learning techniques"
             "for improved performance."
            ),
            "params": 55873736,
            "path": "inception",
        },
     "kaggle_handle": 
     "kaggle://keras/inception_resnet/keras/"
     "inception_resnet_v2_transfer_imagenet/1",
    },
}