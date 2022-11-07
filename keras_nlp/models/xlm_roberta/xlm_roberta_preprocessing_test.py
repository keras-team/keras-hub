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

"""Tests for XLM-RoBERTa preprocessing layers."""

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
            ["the quick brown fox", "the earth is round"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=10,
            model_type="WORD",
            unk_id=0,
            bos_id=1,
            eos_id=2,
        )
        self.proto = bytes_io.getvalue()

        self.preprocessor = XLMRobertaPreprocessor(
            proto=self.proto,
            sequence_length=12,
        )

    def test_tokenize(self):
        input_data = ["the quick brown fox"]

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"], [0, 4, 9, 5, 7, 2, 1, 1, 1, 1, 1, 1]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        )

    def test_unk_token(self):
        input_data = ["the quick brown fox running"]

        output = self.preprocessor(input_data)
        print(output)
        self.assertAllEqual(
            output["token_ids"], [0, 4, 9, 5, 7, 3, 2, 1, 1, 1, 1, 1]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        )

    def test_tokenize_batch(self):
        input_data = tf.constant(
            [
                "the quick brown fox",
                "the quick brown fox",
                "the quick brown fox",
                "the quick brown fox",
            ]
        )

        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"],
            [[0, 4, 9, 5, 7, 2, 1, 1, 1, 1, 1, 1]] * 4,
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] * 4
        )

    def test_tokenize_multiple_sentences(self):
        sentence_one = "the quick brown fox"
        sentence_two = "the earth"

        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"], [0, 4, 9, 5, 7, 2, 2, 4, 6, 2, 1, 1]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        )

    def test_tokenize_multiple_batched_sentences(self):
        sentence_one = tf.constant(
            [
                "the quick brown fox",
                "the quick brown fox",
                "the quick brown fox",
                "the quick brown fox",
            ]
        )
        sentence_two = tf.constant(
            [
                "the earth",
                "the earth",
                "the earth",
                "the earth",
            ]
        )

        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"],
            [[0, 4, 9, 5, 7, 2, 2, 4, 6, 2, 1, 1]] * 4,
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 4
        )

    def test_detokenize(self):
        input_data = tf.constant([[0, 4, 9, 5, 7, 2]])

        output = self.preprocessor.tokenizer.detokenize(input_data)
        self.assertEqual(output, tf.constant(["the quick brown fox"]))

    def test_vocabulary(self):
        vocabulary = self.preprocessor.tokenizer.get_vocabulary()
        self.assertAllEqual(
            vocabulary,
            [
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "▁the",
                "▁brown",
                "▁earth",
                "▁fox",
                "▁is",
                "▁quick",
                "▁round",
            ],
        )
        self.assertEqual(self.preprocessor.vocabulary_size(), 11)

    def test_id_to_token(self):
        print(self.preprocessor.tokenizer.id_to_token(9))
        self.assertEqual(self.preprocessor.tokenizer.id_to_token(9), "▁quick")
        self.assertEqual(self.preprocessor.tokenizer.id_to_token(5), "▁brown")

    def test_token_to_id(self):
        self.assertEqual(self.preprocessor.tokenizer.token_to_id("▁the"), 4)
        self.assertEqual(self.preprocessor.tokenizer.token_to_id("▁round"), 10)
