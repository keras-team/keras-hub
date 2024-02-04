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

import pytest

from keras_nlp.backend import ops
from keras_nlp.models.bloom.bloom_backbone import BloomBackbone
from keras_nlp.tests.test_case import TestCase


class BloomBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 8,
            "intermediate_dim": 32,
            "max_sequence_length": 10,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=BloomBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BloomBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BloomBackbone,
            preset="bloom_560m_multi",
            input_data={
                "token_ids": ops.array([[101, 1996, 4248, 102]], dtype="int32"),
                "padding_mask": ops.ones((1, 4), dtype="int32"),
            },
            expected_output_shape=(1, 4, 1024),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [2.4394186, 1.4131186, -2.7810357, -6.330823, -1.0599766]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BloomBackbone.presets:
            self.run_preset_test(
                cls=BloomBackbone,
                preset=preset,
                input_data=self.input_data,
            )
