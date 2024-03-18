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
import tensorflow as tf

from keras_nlp.models.bert.bert_tokenizer import BertTokenizer
from keras_nlp.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_nlp.tests.test_case import TestCase
from keras_nlp.tokenizers.tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    __test__ = False  # for pytest

    def tokenize(self, inputs):
        return tf.strings.split(inputs).to_tensor()

    def detokenize(self, inputs):
        return tf.strings.reduce_join([inputs], separator=" ", axis=-1)


class TokenizerTest(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTokenizer.presets.keys())
        gpt2_presets = set(GPT2Tokenizer.presets.keys())
        all_presets = set(Tokenizer.presets.keys())
        self.assertContainsSubset(bert_presets, all_presets)
        self.assertContainsSubset(gpt2_presets, all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            Tokenizer.from_preset("bert_tiny_en_uncased"),
            BertTokenizer,
        )
        self.assertIsInstance(
            Tokenizer.from_preset("gpt2_base_en"),
            GPT2Tokenizer,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            GPT2Tokenizer.from_preset("bert_tiny_en_uncased")

    def test_tokenize(self):
        input_data = ["the quick brown fox"]
        tokenizer = SimpleTokenizer()
        tokenize_output = tokenizer.tokenize(input_data)
        call_output = tokenizer(input_data)
        self.assertAllEqual(tokenize_output, [["the", "quick", "brown", "fox"]])
        self.assertAllEqual(call_output, [["the", "quick", "brown", "fox"]])

    def test_detokenize(self):
        input_data = ["the", "quick", "brown", "fox"]
        tokenizer = SimpleTokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["the quick brown fox"])

    def test_missing_tokenize_raises(self):
        with self.assertRaises(NotImplementedError):
            Tokenizer()(["the quick brown fox"])
