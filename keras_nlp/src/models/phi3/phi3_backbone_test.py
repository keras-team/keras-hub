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

from keras_nlp.src.backend import ops
from keras_nlp.src.models.phi3.phi3_backbone import Phi3Backbone
from keras_nlp.src.tests.test_case import TestCase


class Phi3Test(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 8,
        }
        self.su_rotary_init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 12,
            "max_sequence_length": 10,
            "original_max_sequence_length": 5,
            "rope_scaling_type": "su",
            "rope_scaling_short_factor": [1.2, 1.4],
            "rope_scaling_long_factor": [0.8, 0.6],
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Phi3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Phi3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_backbone_basics_with_su_rotary(self):
        self.run_backbone_test(
            cls=Phi3Backbone,
            init_kwargs=self.su_rotary_init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model_with_su_rotary(self):
        self.run_model_saving_test(
            cls=Phi3Backbone,
            init_kwargs=self.su_rotary_init_kwargs,
            input_data=self.input_data,
        )
