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
from keras import ops

from keras_nlp.src.models.bert.bert_backbone import BertBackbone
from keras_nlp.src.tests.test_case import TestCase


class BertBackboneTest(TestCase):
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
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "segment_ids": ops.zeros((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=BertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "sequence_output": (2, 5, 2),
                "pooled_output": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=BertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BertBackbone,
            preset="bert_tiny_en_uncased",
            input_data={
                "token_ids": ops.array([[101, 1996, 4248, 102]], dtype="int32"),
                "segment_ids": ops.zeros((1, 4), dtype="int32"),
                "padding_mask": ops.ones((1, 4), dtype="int32"),
            },
            expected_output_shape={
                "sequence_output": (1, 4, 128),
                "pooled_output": (1, 128),
            },
            # The forward pass from a preset should be stable!
            expected_partial_output={
                "sequence_output": (
                    ops.array([-1.38173, 0.16598, -2.92788, -2.66958, -0.61556])
                ),
                "pooled_output": (
                    ops.array([-0.99999, 0.07777, -0.99955, -0.00982, -0.99967])
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BertBackbone.presets:
            self.run_preset_test(
                cls=BertBackbone,
                preset=preset,
                input_data=self.input_data,
            )
