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
            "pooling": "avg",
        }
        self.input_size = 64
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    @parameterized.named_parameters(
        ("v1_basic", False, "basic_block"),
        ("v1_bottleneck", False, "bottleneck_block"),
        ("v2_basic", True, "basic_block"),
        ("v2_bottleneck", True, "bottleneck_block"),
        ("vd_basic", False, "basic_block_vd"),
        ("vd_bottleneck", False, "bottleneck_block_vd"),
    )
    def test_backbone_basics(self, use_pre_activation, block_type):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {
                "block_type": block_type,
                "use_pre_activation": use_pre_activation,
            }
        )
        if block_type in ("basic_block_vd", "bottleneck_block_vd"):
            init_kwargs.update(
                {
                    "input_conv_filters": [32, 32, 64],
                    "input_conv_kernel_sizes": [3, 3, 3],
                }
            )
        self.run_vision_backbone_test(
            cls=ResNetBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                (2, 64)
                if block_type in ("basic_block", "basic_block_vd")
                else (2, 256)
            ),
            expected_pyramid_output_keys=["P2", "P3", "P4"],
            expected_pyramid_image_sizes=[(16, 16), (8, 8), (4, 4)],
        )

    @parameterized.named_parameters(
        ("v1_basic", False, "basic_block"),
        ("v1_bottleneck", False, "bottleneck_block"),
        ("v2_basic", True, "basic_block"),
        ("v2_bottleneck", True, "bottleneck_block"),
        ("vd_basic", False, "basic_block_vd"),
        ("vd_bottleneck", False, "bottleneck_block_vd"),
    )
    @pytest.mark.large
    def test_saved_model(self, use_pre_activation, block_type):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {
                "block_type": block_type,
                "use_pre_activation": use_pre_activation,
                "image_shape": (None, None, 3),
            }
        )
        if block_type in ("basic_block_vd", "bottleneck_block_vd"):
            init_kwargs.update(
                {
                    "input_conv_filters": [32, 32, 64],
                    "input_conv_kernel_sizes": [3, 3, 3],
                }
            )
        self.run_model_saving_test(
            cls=ResNetBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
        )
