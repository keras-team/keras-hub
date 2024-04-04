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

from keras_nlp.models.bert.bert_masked_lm_preprocessor import (
    BertMaskedLMPreprocessor,
)
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.tests.test_case import TestCase


class TestTask(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertPreprocessor.presets.keys())
        gpt2_presets = set(GPT2Preprocessor.presets.keys())
        all_presets = set(Preprocessor.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            BertPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertPreprocessor,
        )
        self.assertIsInstance(
            BertMaskedLMPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertMaskedLMPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on a preprocessor directly (it is ambiguous).
            Preprocessor.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BertPreprocessor.from_preset("gpt2_base_en")
