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
from keras import backend
from keras import ops

from keras_nlp.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_nlp.src.tests.test_case import TestCase


class ResNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_num_filters": [64, 64, 64],
            "stackwise_num_blocks": [2, 2, 2],
            "stackwise_num_strides": [1, 2, 2],
            "include_rescaling": False,
            "pooling": "avg",
        }
        self.input_size = (16, 16)
        self.input_data = ops.ones((2, 16, 16, 3))

    @parameterized.named_parameters(
        ("v1_basic_channels_last", False, "basic_block", "channels_last"),
        ("v1_basic_channels_first", False, "basic_block", "channels_first"),
        ("v1_block_channels_last", False, "block", "channels_last"),
        ("v2_basic_channels_last", True, "basic_block", "channels_last"),
        ("v2_basic_channels_first", True, "basic_block", "channels_first"),
        ("v2_block_channels_last", True, "block", "channels_last"),
    )
    def test_backbone_basics(self, preact, block_type, data_format):
        if (
            backend.backend() == "tensorflow"
            and data_format == "channels_first"
        ):
            self.skipTest("TensorFlow CPU does not support channels_first")

        if data_format == "channels_last":
            input_data = self.input_data
            input_image_shape = self.input_size + (3,)
        else:
            input_data = ops.transpose(self.input_data, [0, 3, 1, 2])
            input_image_shape = (3,) + self.input_size

        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {
                "block_type": block_type,
                "preact": preact,
                "input_image_shape": input_image_shape,
                "data_format": data_format,
            }
        )
        self.run_backbone_test(
            cls=ResNetBackbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(
                (2, 64) if block_type == "basic_block" else (2, 256)
            ),
            run_quantization_check=(
                True if data_format == "channels_last" else False
            ),
        )

    @parameterized.named_parameters(
        ("v1_basic", False, "basic_block"),
        ("v1_block", False, "block"),
        ("v2_basic", True, "basic_block"),
        ("v2_block", True, "block"),
    )
    @pytest.mark.large
    def test_saved_model(self, preact, block_type):
        init_kwargs = self.init_kwargs.copy()
        init_kwargs.update(
            {
                "block_type": block_type,
                "preact": preact,
                "input_image_shape": (16, 16, 3),
            }
        )
        self.run_model_saving_test(
            cls=ResNetBackbone,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
        )
