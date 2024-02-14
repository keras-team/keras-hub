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

import os

import pytest

from keras_nlp.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_nlp.tests.test_case import TestCase


class MistralTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_mistral_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "mistral_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MistralTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[3, 8, 4, 6], [3, 5, 7, 9]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MistralTokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MistralTokenizer,
            preset="mistral_7b_en",
            input_data=["The quick brown fox."],
            expected_output=[[415, 2936, 9060, 285, 1142, 28723]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralTokenizer.presets:
            self.run_preset_test(
                cls=MistralTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
