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

import numpy as np
import pytest

from keras_nlp.src.models.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)
from keras_nlp.src.tests.test_case import TestCase


class MobileNetV3BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "stackwise_expansion": [1, 4, 6],
            "stackwise_filters": [4, 8, 16],
            "stackwise_kernel_size": [3, 3, 5],
            "stackwise_stride": [2, 2, 1],
            "stackwise_se_ratio": [0.25, None, 0.25],
            "stackwise_activation": ["relu", "relu", "hard_swish"],
            "include_rescaling": False,
            "input_shape": (224, 224, 3),
            "alpha": 1,
        }
        self.input_data = np.ones((2, 224, 224, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MobileNetV3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 28, 28, 96),
            run_mixed_precision_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetV3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
