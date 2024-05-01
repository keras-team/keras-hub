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

from keras_nlp.src.backend import ops
from keras_nlp.src.models.f_net.f_net_backbone import FNetBackbone
from keras_nlp.src.tests.test_case import TestCase


class FNetBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
            "max_sequence_length": 5,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "segment_ids": ops.zeros((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=FNetBackbone,
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
            cls=FNetBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=FNetBackbone,
            preset="f_net_base_en",
            input_data={
                "token_ids": ops.array([[101, 1996, 4248, 102]], dtype="int32"),
                "segment_ids": ops.zeros((1, 4), dtype="int32"),
            },
            expected_output_shape={
                "sequence_output": (1, 4, 768),
                "pooled_output": (1, 768),
            },
            # The forward pass from a preset should be stable!
            expected_partial_output={
                "sequence_output": (
                    ops.array([4.15728, -0.09661, -0.24494, -0.06810, -0.55959])
                ),
                "pooled_output": (
                    ops.array([-0.04117, -0.03273, -0.02134, 0.99754, -0.09777])
                ),
            },
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in FNetBackbone.presets:
            self.run_preset_test(
                cls=FNetBackbone,
                preset=preset,
                input_data=self.input_data,
            )
