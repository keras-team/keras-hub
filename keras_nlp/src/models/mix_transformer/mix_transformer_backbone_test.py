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
from keras import models

from keras_nlp.src.models.mix_transformer.mix_transformer_backbone import (
    MiTBackbone,
)
from keras_nlp.src.tests.test_case import TestCase


class MiTBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "depths": [2, 2],
            "include_rescaling": True,
            "image_shape": (16, 16, 3),
            "hidden_dims": [4, 8],
            "num_layers": 2,
            "blockwise_num_heads": [1, 2],
            "blockwise_sr_ratios": [8, 4],
            "end_value": 0.1,
            "patch_sizes": [7, 3],
            "strides": [4, 2],
        }
        self.input_size = 16
        self.input_data = np.ones(
            (2, self.input_size, self.input_size, 3), dtype="float32"
        )

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=MiTBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 2, 2, 8),
            run_quantization_check=False,
            run_mixed_precision_check=False,
        )

    def test_pyramid_output_format(self):
        init_kwargs = self.init_kwargs
        backbone = MiTBackbone(**init_kwargs)
        model = models.Model(backbone.inputs, backbone.pyramid_outputs)
        output_data = model(self.input_data)

        self.assertIsInstance(output_data, dict)
        self.assertEqual(
            list(output_data.keys()), list(backbone.pyramid_outputs.keys())
        )
        self.assertEqual(list(output_data.keys()), ["P1", "P2"])
        for k, v in output_data.items():
            size = self.input_size // (2 ** (int(k[1:]) + 1))
            self.assertEqual(tuple(v.shape[:3]), (2, size, size))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MiTBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
