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

from keras_nlp.src.models.bert.bert_text_classifier_preprocessor import (
    BertTextClassifierPreprocessor,
)
from keras_nlp.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_nlp.src.models.text_classifier_preprocessor import (
    TextClassifierPreprocessor,
)
from keras_nlp.src.tests.test_case import TestCase


class TestTextClassifierPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTextClassifierPreprocessor.presets.keys())
        gpt2_presets = set(GPT2Tokenizer.presets.keys())
        all_presets = set(TextClassifierPreprocessor.presets.keys())
        self.assertTrue(bert_presets.issubset(all_presets))
        self.assertTrue(gpt2_presets.isdisjoint(all_presets))

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            TextClassifierPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertTextClassifierPreprocessor,
        )
        self.assertIsInstance(
            BertTextClassifierPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertTextClassifierPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = TextClassifierPreprocessor.from_preset(
            "bert_tiny_en_uncased", sequence_length=16
        )
        self.assertEqual(preprocessor.sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BertTextClassifierPreprocessor.from_preset("gpt2_base_en")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            BertTextClassifierPreprocessor.from_preset(
                "hf://spacy/en_core_web_sm"
            )
