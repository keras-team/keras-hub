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

from keras_nlp.src.models.vit_det.vit_det_backbone import ViTDetBackbone
from keras_nlp.src.tests.test_case import TestCase


class ViTDetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "include_rescaling": True,
            "input_shape": (16, 16, 3),
            "patch_size": 2,
            "embedding_dim": 4,
            "depth": 2,
            "mlp_dim": 4 * 4,
            "num_heads": 2,
            "num_output_channels": 2,
            "window_size": 2,
        }
        self.input_data = np.ones((1, 16, 16, 3), dtype="float32")

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=ViTDetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 8, 8, 2),
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ViTDetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
