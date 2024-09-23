# Copyright 2024 The KerasHub Authors
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
from keras import ops

from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class PaliGemmaBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 16
        self.image_sequence_length = int((self.image_size / 4) ** 2)
        self.init_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "image_size": self.image_size,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "vit_patch_size": 4,
            "vit_num_layers": 2,
            "vit_num_heads": 2,
            "vit_hidden_dim": 8,
            "vit_intermediate_dim": 16,
        }

        dummy_images = np.random.rand(
            self.batch_size, self.image_size, self.image_size, 3
        )
        dummy_text_token_ids = np.random.rand(
            self.batch_size, self.text_sequence_length
        )
        self.input_data = {
            "token_ids": dummy_text_token_ids,
            "images": dummy_images,
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "response_mask": np.zeros(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=PaliGemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length + self.image_sequence_length,
                8,
            ),
            variable_length_data=[self.input_data],
            run_mixed_precision_check=False,  # TODO: Set to `True`
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=PaliGemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=PaliGemmaBackbone,
            preset="pali_gemma_3b_mix_224",
            input_data={
                "token_ids": ops.array([[1169, 2068, 7586, 21831, 13]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
                "response_mask": ops.zeros((1, 5), dtype="int32"),
                "images": ops.zeros((1, 224, 224, 3), dtype="float32"),
            },
            expected_output_shape=(1, 261, 2048),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [-0.449851, 1.431027, -0.713446, 0.417485, -0.640859]
            ),
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in PaliGemmaBackbone.presets:
            self.run_preset_test(
                cls=PaliGemmaBackbone,
                preset=preset,
                input_data=self.input_data,
            )
