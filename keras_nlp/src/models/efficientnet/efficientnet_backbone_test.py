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

import keras
import pytest
from absl.testing import parameterized

from keras_nlp.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)
from keras_nlp.src.tests.test_case import TestCase


class EfficientNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [2, 4, 4, 6, 9, 15],
            "stackwise_input_filters": [24, 24, 48, 64, 128, 160],
            "stackwise_output_filters": [24, 48, 64, 128, 160, 256],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [
                0.0,
                0.0,
                0,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_block_types": ["fused"] * 3 + ["unfused"] * 3,
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "include_rescaling": False,
        }
        self.input_data = keras.ops.ones(shape=(8, 224, 224, 3))

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=EfficientNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            run_mixed_precision_check=False,
            expected_output_shape=(8, 7, 7, 1280),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=EfficientNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_valid_call(self):
        model = EfficientNetBackbone(**self.init_kwargs)
        model(self.input_data)

    def test_valid_call_original_v1(self):
        original_v1_kwargs = {
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "include_rescaling": False,
            "stackwise_block_types": ["v1"] * 7,
            "min_depth": None,
            "include_initial_padding": True,
            "use_depth_divisor_as_min_depth": True,
            "cap_round_filter_decrease": True,
            "stem_conv_padding": "valid",
            "batch_norm_momentum": 0.99,
        }
        model = EfficientNetBackbone(**original_v1_kwargs)
        model(self.input_data)

    def test_valid_call_with_rescaling(self):
        test_kwargs = self.init_kwargs.copy()
        test_kwargs["include_rescaling"] = True
        model = EfficientNetBackbone(**test_kwargs)
        model(self.input_data)

    def test_feature_pyramid_outputs(self):
        backbone = EfficientNetBackbone(**self.init_kwargs)
        model = keras.Model(
            inputs=backbone.inputs, outputs=backbone.pyramid_outputs
        )
        batch_size = 8
        height = width = 256
        outputs = model(keras.ops.ones(shape=(batch_size, height, width, 3)))
        levels = ["P1", "P2", "P3", "P4", "P5"]
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(
            outputs["P1"].shape,
            (batch_size, height // 2**1, width // 2**1, 24),
        )
        self.assertEquals(
            outputs["P2"].shape,
            (batch_size, height // 2**2, width // 2**2, 48),
        )
        self.assertEquals(
            outputs["P3"].shape,
            (batch_size, height // 2**3, width // 2**3, 64),
        )
        self.assertEquals(
            outputs["P4"].shape,
            (batch_size, height // 2**4, width // 2**4, 160),
        )
        self.assertEquals(
            outputs["P5"].shape,
            (batch_size, height // 2**5, width // 2**5, 1280),
        )

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        test_kwargs = self.init_kwargs.copy()
        test_kwargs["input_shape"] = (None, None, num_channels)
        model = EfficientNetBackbone(**test_kwargs)
        self.assertEqual(model.output_shape, (None, None, None, 1280))
