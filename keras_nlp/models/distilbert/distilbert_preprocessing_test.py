# Copyright 2022 The KerasNLP Authors
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
"""Tests for DistilBERT preprocessing layers."""

import tensorflow as tf

from keras_nlp.models.distilbert.distilbert_preprocessing import (
    DistilBertPreprocessor,
)


class DistilBertPreprocessorTest(tf.test.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]

    def test_tokenize(self):
        input_data = ["THE QUICK BROWN FOX."]
        preprocessor = DistilBertPreprocessor(
            vocabulary=self.vocab,
            sequence_length=8,
        )
        output = preprocessor(input_data)
        self.assertAllEqual(output["token_ids"], [2, 5, 6, 7, 8, 1, 3, 0])
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0])

    def test_lowercase(self):
        input_data = ["THE QUICK BROWN FOX."]
        preprocessor = DistilBertPreprocessor(
            vocabulary=self.vocab,
            sequence_length=8,
            lowercase=True,
        )
        output = preprocessor(input_data)
        self.assertAllEqual(output["token_ids"], [2, 9, 10, 11, 12, 1, 3, 0])

    def test_detokenize(self):
        input_data = [[5, 6, 7, 8]]
        preprocessor = DistilBertPreprocessor(vocabulary=self.vocab)
        output = preprocessor.tokenizer.detokenize(input_data)
        self.assertAllEqual(output, ["THE QUICK BROWN FOX"])

    def test_vocabulary_size(self):
        preprocessor = DistilBertPreprocessor(vocabulary=self.vocab)
        self.assertEqual(preprocessor.vocabulary_size(), 13)
