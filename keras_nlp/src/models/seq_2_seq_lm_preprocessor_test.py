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

from keras_nlp.src.models.bart.bart_preprocessor import BartPreprocessor
from keras_nlp.src.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_nlp.src.models.bert.bert_tokenizer import BertTokenizer
from keras_nlp.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_nlp.src.tests.test_case import TestCase


class TestSeq2SeqLMPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTokenizer.presets.keys())
        bart_presets = set(BartPreprocessor.presets.keys())
        all_presets = set(Seq2SeqLMPreprocessor.presets.keys())
        self.assertTrue(bert_presets.isdisjoint(all_presets))
        self.assertTrue(bart_presets.issubset(all_presets))

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            Seq2SeqLMPreprocessor.from_preset("bart_base_en"),
            BartSeq2SeqLMPreprocessor,
        )
        self.assertIsInstance(
            BartSeq2SeqLMPreprocessor.from_preset("bart_base_en"),
            BartSeq2SeqLMPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = Seq2SeqLMPreprocessor.from_preset(
            "bart_base_en", decoder_sequence_length=16
        )
        self.assertEqual(preprocessor.decoder_sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BartSeq2SeqLMPreprocessor.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            BartSeq2SeqLMPreprocessor.from_preset("hf://spacy/en_core_web_sm")
