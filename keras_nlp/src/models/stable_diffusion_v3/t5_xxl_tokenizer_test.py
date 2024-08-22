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
import os

import pytest

from keras_nlp.src.models.stable_diffusion_v3.t5_xxl_tokenizer import (
    T5XXLTokenizer,
)
from keras_nlp.src.tests.test_case import TestCase


class T5XXLTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_t5_test_proto.py
            "proto": os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=T5XXLTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            # </s> (id: 1) will be added to the output.
            expected_output=[[4, 9, 5, 7, 1], [4, 6, 8, 10, 1]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            T5XXLTokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.skipTest("TODO: Add preset from `hf://google/t5-v1_1-xxl`")
        for preset in T5XXLTokenizer.presets:
            self.run_preset_test(
                cls=T5XXLTokenizer,
                preset=preset,
                input_data=["The quick brown fox."],
                expected_output=[[37, 1704, 4216, 3, 20400, 5]],
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        self.skipTest("TODO: Add preset from `hf://google/t5-v1_1-xxl`")
        for preset in T5XXLTokenizer.presets:
            self.run_preset_test(
                cls=T5XXLTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
