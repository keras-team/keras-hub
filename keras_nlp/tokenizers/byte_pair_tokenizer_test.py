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

import tensorflow as tf

from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer


class BytePairTokenizerTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.vocabulary = {
            "t": 1,
            "h": 2,
            "e": 3,
            " ": 4,
            "the": 5,
            "b": 6,
            "r": 7,
            "o": 8,
            "w": 9,
            "n": 10,
            "brown": 11,
            ".": 12,
        }

    def test_tokenize(self):
        input_data = ["brown."]
        tokenizer = BytePairTokenizer(
            vocabulary=self.vocabulary,
            merges=["b r", "br o", "bro w", "brow n"],
        )
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertIsInstance(call_output, tf.RaggedTensor)
        self.assertAllEqual(call_output, [[11, 12]])
        self.assertAllEqual(tokenize_output, [[11, 12]])

    def test_tokenize_scalar(self):
        input_data = "brown."
        tokenizer = BytePairTokenizer(
            vocabulary=self.vocabulary,
            merges=["b r", "br o", "bro w", "brow n"],
        )
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertAllEqual(tokenize_output, [11, 12])

    def test_tokenize_single_output(self):
        # Test that output doesn't collapse to zero dimensions with one output
        input_data = "brown"
        tokenizer = BytePairTokenizer(
            vocabulary=self.vocabulary,
            merges=["b r", "br o", "bro w", "brow n"],
        )
        tokenize_output = tokenizer.tokenize(input_data)
        self.assertAllEqual(tokenize_output, [11])

    def test_detokenize(self):
        input_data = ["brown."]
        tokenizer = BytePairTokenizer(
            vocabulary=self.vocabulary,
            merges=["b r", "br o", "bro w", "brow n"],
        )
        tokenized_data = tokenizer.tokenize(input_data)
        output_data = tokenizer.detokenize(tokenized_data)
        self.assertAllEqual(input_data, output_data)
