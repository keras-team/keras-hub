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

from keras_hub.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_hub.src.tests.test_case import TestCase


class MobileNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_expansion": [1, 4, 6],
            "stackwise_num_filters": [4, 8, 16],
            "stackwise_kernel_size": [3, 3, 5],
            "stackwise_num_strides": [2, 2, 1],
            "stackwise_se_ratio": [0.25, None, 0.25],
            "stackwise_activation": ["relu", "relu", "hard_swish"],
            "output_num_filters": 1280,
            "input_activation": "hard_swish",
            "output_activation": "hard_swish",
            "inverted_res_block": True,
            "input_num_filters": 16,
            "image_shape": (224, 224, 3),
            "depth_multiplier": 1,
        }
        self.input_data = np.ones((2, 224, 224, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_vision_backbone_test(
            cls=MobileNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 28, 28, 96),
            run_mixed_precision_check=False,
            run_data_format_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
