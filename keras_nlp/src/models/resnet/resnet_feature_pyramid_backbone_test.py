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

from absl.testing import parameterized
from keras import ops

from keras_nlp.src.models.resnet.resnet_feature_pyramid_backbone import (
    ResNetFeaturePyramidBackbone,
)
from keras_nlp.src.tests.test_case import TestCase


class ResNetFeaturePyramidBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "input_image_shape": (None, None, 3),
            "pooling": "avg",
        }
        self.input_size = 32
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

    @parameterized.named_parameters(
        ("v1_basic", False, "basic_block"),
        ("v1_bottleneck", False, "bottleneck_block"),
        ("v2_basic", True, "basic_block"),
        ("v2_bottleneck", True, "bottleneck_block"),
    )
    def test_pyramid_outputs(self, use_pre_activation, block_type):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {"block_type": block_type, "use_pre_activation": use_pre_activation}
        )
        model = ResNetFeaturePyramidBackbone(**init_kwargs)
        output_data = model(self.input_data)

        self.assertIsInstance(output_data, dict)
        self.assertEqual(list(model.pyramid_outputs.keys()), ["P2", "P3", "P4"])
        self.assertEqual(
            list(output_data.keys()), list(model.pyramid_outputs.keys())
        )
        for k, v in output_data.items():
            size = self.input_size // (2 ** int(k[1:]))
            self.assertEqual(tuple(v.shape[:3]), (2, size, size))

    def test_output_keys(self):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {"block_type": "basic_block", "use_pre_activation": False}
        )
        model = ResNetFeaturePyramidBackbone(
            **init_kwargs, output_keys=["P3", "P4"]
        )
        output_data = model(self.input_data)

        self.assertIsInstance(output_data, dict)
        self.assertEqual(list(output_data.keys()), ["P3", "P4"])
