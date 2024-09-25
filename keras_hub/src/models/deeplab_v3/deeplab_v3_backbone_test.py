# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from keras_hub.src.models.deeplab_v3.deeplab_v3_backbone import (
    DeepLabV3Backbone,
)
from keras_hub.src.models.deeplab_v3.deeplab_v3_layers import (
    SpatialPyramidPooling,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DeepLabV3Test(TestCase):
    def setUp(self):
        self.resnet_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "block_type": "basic_block",
            "use_pre_activation": False,
        }
        self.image_encoder = ResNetBackbone(**self.resnet_kwargs)
        self.init_kwargs = {
            "image_encoder": self.image_encoder,
            "low_level_feature_key": "P2",
            "spatial_pyramid_pooling_key": "P4",
            "dilation_rates": [6, 12, 18],
            "upsampling_size": 4,
            "image_shape": (96, 96, 3),
        }
        self.input_data = np.ones((2, 96, 96, 3), dtype="float32")

    def test_segmentation_basics(self):
        self.run_vision_backbone_test(
            cls=DeepLabV3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 96, 96, 256),
            run_mixed_precision_check=False,
            run_quantization_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DeepLabV3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )


class SpatialPyramidPoolingTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=SpatialPyramidPooling,
            init_kwargs={
                "dilation_rates": [6, 12, 18],
                "activation": "relu",
                "num_channels": 256,
                "dropout": 0.1,
            },
            input_data=np.random.randn(1, 4, 4, 6),
            expected_output_shape=(1, 4, 4, 256),
            expected_num_trainable_weights=18,
            expected_num_non_trainable_variables=13,
            expected_num_non_trainable_weights=12,
        )
