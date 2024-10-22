# Copyright 2024 The KerasNLP Authors
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

import pytest
from keras import ops

from keras_hub.src.models.differential_binarization.differential_binarization import (
    DifferentialBinarization,
)
from keras_hub.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_hub.src.tests.test_case import TestCase


class DifferentialBinarizationTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 224, 224, 3))
        self.labels = ops.zeros((2, 224, 224, 4))
        self.backbone = ResNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 128, 256, 512],
            stackwise_num_blocks=[3, 4, 6, 3],
            stackwise_num_strides=[1, 2, 2, 2],
            block_type="bottleneck_block",
            image_shape=(224, 224, 3),
            include_rescaling=False,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
        }
        self.train_data = (self.images, self.labels)

    def test_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=DifferentialBinarization,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 224, 224, 3),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DifferentialBinarization,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
