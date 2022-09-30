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
"""Tests for BERT preprocessing layers."""

import io

import sentencepiece
import tensorflow as tf

from keras_nlp.models.xlm_roberta.xlm_roberta_preprocessing import (
    XLMRobertaPreprocessor,
)


class XLMRobertaPreprocessorTest(tf.test.TestCase):
    def setUp(self):
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox."]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=7,
            model_type="WORD",
        )
        self.proto = bytes_io.getvalue()

    def test_tokenize(self):
        input_data = ["the quick brown fox."]
        preprocessor = XLMRobertaPreprocessor(
            proto=self.proto,
            sequence_length=8,
        )
        output = preprocessor(input_data)

        self.assertAllEqual(output["token_ids"], [0, 7, 6, 4, 5, 2, 1, 1])
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 1, 0, 0])

    def test_vocabulary_size(self):
        preprocessor = XLMRobertaPreprocessor(proto=self.proto)
        self.assertEqual(preprocessor.vocabulary_size(), 7)
