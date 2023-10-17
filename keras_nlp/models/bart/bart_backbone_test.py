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
from keras_nlp.models.bart.bart_backbone import BartBackbone
from keras_nlp.tests.test_case import TestCase


class BartBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "encoder_token_ids": ops.ones((2, 3), dtype="int32"),
            "encoder_padding_mask": ops.zeros((2, 3), dtype="int32"),
            "decoder_token_ids": ops.ones((2, 5), dtype="int32"),
            "decoder_padding_mask": ops.zeros((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=BartBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 3, 2),
                "decoder_sequence_output": (2, 5, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BartBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BartBackbone,
            preset="bart_base_en",
            input_data={
                "encoder_token_ids": ops.array([[0, 133, 2119, 2]]),
                "encoder_padding_mask": ops.array([[1, 1, 1, 1]]),
                "decoder_token_ids": ops.array([[0, 7199, 14, 2119, 2]]),
                "decoder_padding_mask": ops.array([[1, 1, 1, 1, 1]]),
            },
            expected_output_shape={
                "encoder_sequence_output": (1, 4, 768),
                "decoder_sequence_output": (1, 5, 768),
            },
            # The forward pass from a preset should be stable!
            expected_partial_output={
                "encoder_sequence_output": ops.array(
                    [-0.033, 0.013, -0.003, -0.012, -0.002]
                ),
                "decoder_sequence_output": ops.array(
                    [2.516, 2.489, 0.695, 8.057, 1.245]
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BartBackbone.presets:
            self.run_preset_test(
                cls=BartBackbone,
                preset=preset,
                input_data=self.input_data,
            )
