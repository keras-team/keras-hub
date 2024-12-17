#!/usr/bin/env python3

"""Converts BASNet model weights to KerasHub.

Usage: python3 convert_basnet_checkpoint.py

Downloads BASNet modelweights with ResNet34 backbone and converts them to a
KerasHub model. Credits for model training go to Hamid Ali
(https://github.com/hamidriasat/BASNet).

Requirements:
pip3 install -q git+https://github.com/keras-team/keras-hub.git
pip3 install -q gdown
"""

import gdown

from keras_hub.src.models.basnet.basnet import BASNetImageSegmenter
from keras_hub.src.models.basnet.basnet_backbone import BASNetBackbone
from keras_hub.src.models.basnet.basnet_preprocessor import BASNetPreprocessor
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone

# download weights
gdown.download(
    "https://drive.google.com/uc?id=1OWKouuAQ7XpXZbWA3mmxDPrFGW71Axrg",
    "basnet_weights.h5",
)

# instantiate ResNet34
image_encoder = ResNetBackbone(
    input_conv_filters=[64],
    input_conv_kernel_sizes=[7],
    stackwise_num_filters=[64, 128, 256, 512],
    stackwise_num_blocks=[3, 4, 6, 3],
    stackwise_num_strides=[1, 2, 2, 2],
    block_type="basic_block",
)

# instantiate BASNet and load pretrained weights
preprocessor = BASNetPreprocessor()
backbone = BASNetBackbone(image_encoder=image_encoder, num_classes=1)
basnet = BASNetImageSegmenter(backbone=backbone, preprocessor=preprocessor)
backbone.load_weights("basnet_weights.h5")

# save the preset
basnet.save_to_preset("basnet")