# Copyright 2023 The KerasNLP Authors
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

from keras_nlp.src.models.vgg16.vgg16_backbone import VGG16Backbone
from keras_nlp.src.tests.test_case import TestCase


class VGG16BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "input_shape": (224, 224, 3),
            "include_top": False,
            "include_rescaling": False,
            "pooling": "avg",
        }
        self.input_data = np.ones((2, 224, 224, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_cv_backbone_test(
            cls=VGG16Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 512),
            run_mixed_precision_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VGG16Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
