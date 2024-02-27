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

from keras_nlp.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_nlp.tests.test_case import TestCase


@pytest.mark.keras_3_only
class GemmaTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_gemma_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "gemma_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=GemmaTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[4, 9, 5, 7], [4, 6, 8, 10]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            GemmaTokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.kaggle_key_required
    @pytest.mark.large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=GemmaTokenizer,
            preset="gemma_2b_en",
            input_data=["The quick brown fox."],
            expected_output=[[651, 4320, 8426, 25341, 235265]],
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GemmaTokenizer.presets:
            self.run_preset_test(
                cls=GemmaTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
