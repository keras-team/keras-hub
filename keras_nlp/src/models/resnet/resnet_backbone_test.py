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
from keras import models
from keras import ops

from keras_nlp.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_nlp.src.tests.test_case import TestCase


class ResNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "input_image_shape": (None, None, 3),
            "pooling": "avg",
        }
        self.input_size = 64
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    @parameterized.named_parameters(
        ("v1_basic", False, False, "basic_block"),
        ("v1_bottleneck", False, False, "bottleneck_block"),
        ("v2_basic", True, False, "basic_block"),
        ("v2_bottleneck", True, False, "bottleneck_block"),
        ("vd_basic", False, True, "basic_block"),
        ("vd_bottleneck", False, True, "bottleneck_block"),
    )
    def test_backbone_basics(
        self, use_pre_activation, use_vd_pooling, block_type
    ):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {
                "block_type": block_type,
                "use_pre_activation": use_pre_activation,
                "use_vd_pooling": use_vd_pooling,
            }
        )
        self.run_vision_backbone_test(
            cls=ResNetBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                (2, 64) if block_type == "basic_block" else (2, 256)
            ),
        )

    def test_pyramid_output_format(self):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {"block_type": "basic_block", "use_pre_activation": False}
        )
        backbone = ResNetBackbone(**init_kwargs)
        model = models.Model(backbone.inputs, backbone.pyramid_outputs)
        output_data = model(self.input_data)

        self.assertIsInstance(output_data, dict)
        self.assertEqual(
            list(output_data.keys()), list(backbone.pyramid_outputs.keys())
        )
        self.assertEqual(list(output_data.keys()), ["P2", "P3", "P4"])
        for k, v in output_data.items():
            size = self.input_size // (2 ** int(k[1:]))
            self.assertEqual(tuple(v.shape[:3]), (2, size, size))

    @parameterized.named_parameters(
        ("v1_basic", False, False, "basic_block"),
        ("v1_bottleneck", False, False, "bottleneck_block"),
        ("v2_basic", True, False, "basic_block"),
        ("v2_bottleneck", True, False, "bottleneck_block"),
        ("vd_basic", False, True, "basic_block"),
        ("vd_bottleneck", False, True, "bottleneck_block"),
    )
    @pytest.mark.large
    def test_saved_model(self, use_pre_activation, use_vd_pooling, block_type):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {
                "block_type": block_type,
                "use_pre_activation": use_pre_activation,
                "use_vd_pooling": use_vd_pooling,
                "input_image_shape": (None, None, 3),
            }
        )
        self.run_model_saving_test(
            cls=ResNetBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
        )
