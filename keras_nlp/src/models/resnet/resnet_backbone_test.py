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
from absl.testing import parameterized
from keras import ops

from keras_nlp.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_nlp.src.tests.test_case import TestCase


class ResNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "input_conv_filters": [64],
            "input_conv_kernel_sizes": [7],
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "image_shape": (None, None, 3),
            "block_type": "bottleneck_block",
            "use_pre_activation": False,
        }
        self.input_size = 64
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    @parameterized.named_parameters(
        ("basic", "basic_block"),
        ("bottleneck", "bottleneck_block"),
    )
    def test_backbone_basics(self, block_type):
        feature_size = 64 if block_type == "basic_block" else 256
        self.run_vision_backbone_test(
            cls=ResNetBackbone,
            init_kwargs={**self.init_kwargs, "block_type": block_type},
            input_data=self.input_data,
            expected_output_shape=(2, 4, 4, feature_size),
            expected_pyramid_output_keys=["P2", "P3", "P4"],
            expected_pyramid_image_sizes=[(16, 16), (8, 8), (4, 4)],
        )

    @parameterized.named_parameters(
        ("basic", "basic_block"),
        ("bottleneck", "bottleneck_block"),
    )
    def test_backbone_v2(self, block_type):
        feature_size = 64 if block_type == "basic_block" else 256
        self.run_vision_backbone_test(
            cls=ResNetBackbone,
            init_kwargs={
                **self.init_kwargs,
                "block_type": block_type,
                "use_pre_activation": True,
            },
            input_data=self.input_data,
            expected_output_shape=(2, 4, 4, feature_size),
            expected_pyramid_output_keys=["P2", "P3", "P4"],
            expected_pyramid_image_sizes=[(16, 16), (8, 8), (4, 4)],
        )

    @parameterized.named_parameters(
        ("basic", "basic_block_vd"),
        ("bottleneck", "bottleneck_block_vd"),
    )
    def test_backbone_vd(self, block_type):
        feature_size = 64 if block_type == "basic_block_vd" else 256
        self.run_vision_backbone_test(
            cls=ResNetBackbone,
            init_kwargs={
                **self.init_kwargs,
                "block_type": block_type,
                "input_conv_filters": [32, 32, 64],
                "input_conv_kernel_sizes": [3, 3, 3],
            },
            input_data=self.input_data,
            expected_output_shape=(2, 4, 4, feature_size),
            expected_pyramid_output_keys=["P2", "P3", "P4"],
            expected_pyramid_image_sizes=[(16, 16), (8, 8), (4, 4)],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ResNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ResNetBackbone.presets:
            self.run_preset_test(
                cls=ResNetBackbone,
                preset=preset,
                input_data=self.input_data,
            )
