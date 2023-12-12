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

from keras_nlp.backend import config as backend_config
from keras_nlp.backend import ops
from keras_nlp.models.t5.t5_backbone import T5Backbone
from keras_nlp.tests.test_case import TestCase


class T5BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 2,
            "intermediate_dim": 4,
        }
        self.input_data = {
            "encoder_token_ids": ops.ones((2, 3), dtype="int32"),
            "encoder_padding_mask": ops.zeros((2, 3), dtype="int32"),
            "decoder_token_ids": ops.ones((2, 3), dtype="int32"),
            "decoder_padding_mask": ops.zeros((2, 3), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=T5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 3, 2),
                "decoder_sequence_output": (2, 3, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=T5Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.large
    @pytest.mark.skipif(
        not backend_config.keras_3(),
        reason="TODO: Fails in Keras2",
    )
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=T5Backbone,
            preset="t5_small_multi",
            input_data=self.input_data,
            expected_output_shape={
                "encoder_sequence_output": (2, 3, 512),
                "decoder_sequence_output": (2, 3, 512),
            },
            expected_partial_output={
                "encoder_sequence_output": ops.array(
                    [-0.0034, 0.0293, -0.0827, -0.1076]
                ),
                "decoder_sequence_output": ops.array(
                    [0.0097, 0.3576, -0.1508, 0.0150]
                ),
            },
        )

    @pytest.mark.extra_large
    @pytest.mark.skipif(
        not backend_config.keras_3(),
        reason="TODO: Fails in Keras2",
    )
    def test_all_presets(self):
        for preset in T5Backbone.presets:
            self.run_preset_test(
                cls=T5Backbone,
                preset=preset,
                input_data=self.input_data,
            )
