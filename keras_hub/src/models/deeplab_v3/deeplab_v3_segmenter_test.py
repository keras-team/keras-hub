# Copyright 2024 The KerasHUB Authors
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
from keras_hub.src.models.deeplab_v3.deeplab_v3_segmenter import (
    DeepLabV3ImageSegmenter,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DeepLabV3ImageSegmenterTest(TestCase):
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
        self.deeplab_backbone = DeepLabV3Backbone(
            image_encoder=self.image_encoder,
            low_level_feature_key="P2",
            spatial_pyramid_pooling_key="P4",
            dilation_rates=[6, 12, 18],
            upsampling_size=4,
        )
        self.init_kwargs = {
            "backbone": self.deeplab_backbone,
            "num_classes": 2,
            "activation": "softmax",
        }
        self.images = np.ones((2, 96, 96, 3), dtype="float32")
        self.labels = np.zeros((2, 96, 96, 2), dtype="float32")
        self.train_data = (
            self.images,
            self.labels,
        )

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=DeepLabV3ImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            batch_size=2,
            expected_output_shape=(2, 96, 96, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DeepLabV3ImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
