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

import unittest

import tensorflow as tf

from keras_nlp.models.bert.bert_preprocessing import BertPreprocessor


class BertPreprocessorTest(tf.test.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]

    def test_tokenize(self):
        input_data = ["THE QUICK BROWN FOX."]
        preprocessor = BertPreprocessor(
            vocabulary=self.vocab,
            sequence_length=8,
        )
        output = preprocessor(input_data)
        self.assertAllEqual(output["token_ids"], [2, 5, 6, 7, 8, 1, 3, 0])
        self.assertAllEqual(output["segment_ids"], [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 0])

    def test_tokenize_batch(self):
        input_data = tf.constant(
            [
                "THE QUICK BROWN FOX.",
                "THE QUICK BROWN FOX.",
                "THE QUICK BROWN FOX.",
                "THE QUICK BROWN FOX.",
            ]
        )
        preprocessor = BertPreprocessor(
            vocabulary=self.vocab,
            sequence_length=8,
        )
        output = preprocessor(input_data)
        self.assertAllEqual(output["token_ids"], [[2, 5, 6, 7, 8, 1, 3, 0]] * 4)
        self.assertAllEqual(
            output["segment_ids"], [[0, 0, 0, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4
        )

    def test_tokenize_multiple_sentences(self):
        sentence_one = "THE QUICK"
        sentence_two = "BROWN FOX."
        preprocessor = BertPreprocessor(
            vocabulary=self.vocab,
            sequence_length=8,
        )
        # The first tuple or list is always interpreted as an enumeration of
        # separate sequences to concatenate.
        output = preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(output["token_ids"], [2, 5, 6, 3, 7, 8, 1, 3])
        self.assertAllEqual(output["segment_ids"], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1])

    def test_tokenize_multiple_batched_sentences(self):
        sentence_one = tf.constant(
            [
                "THE QUICK",
                "THE QUICK",
                "THE QUICK",
                "THE QUICK",
            ]
        )
        sentence_two = tf.constant(
            [
                "BROWN FOX.",
                "BROWN FOX.",
                "BROWN FOX.",
                "BROWN FOX.",
            ]
        )
        preprocessor = BertPreprocessor(
            vocabulary=self.vocab,
            sequence_length=8,
        )
        # The first tuple or list is always interpreted as an enumeration of
        # separate sequences to concatenate.
        output = preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(output["token_ids"], [[2, 5, 6, 3, 7, 8, 1, 3]] * 4)
        self.assertAllEqual(
            output["segment_ids"], [[0, 0, 0, 0, 1, 1, 1, 1]] * 4
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1]] * 4
        )

    def test_lowercase(self):
        input_data = ["THE QUICK BROWN FOX."]
        preprocessor = BertPreprocessor(
            vocabulary=self.vocab,
            sequence_length=8,
            lowercase=True,
        )
        output = preprocessor(input_data)
        self.assertAllEqual(output["token_ids"], [2, 9, 10, 11, 12, 1, 3, 0])

    def test_detokenize(self):
        input_tokens = [[5, 6, 7, 8]]
        preprocessor = BertPreprocessor(vocabulary=self.vocab)
        output = preprocessor.tokenizer.detokenize(input_tokens)
        self.assertAllEqual(output, ["THE QUICK BROWN FOX"])

    def test_vocabulary_size(self):
        preprocessor = BertPreprocessor(vocabulary=self.vocab)
        self.assertEqual(preprocessor.vocabulary_size(), 13)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            BertPreprocessor.from_preset("bert_base_uncased_clowntown")

    @unittest.mock.patch("tensorflow.keras.utils.get_file")
    def test_valid_call_presets(self, get_file_mock):
        """Ensure presets have necessary structure, but no RCPs."""
        input_data = ["THE QUICK BROWN FOX."]
        get_file_mock.return_value = self.vocab
        for preset in BertPreprocessor.presets:
            preprocessor = BertPreprocessor.from_preset(preset)
            preprocessor(input_data)
        self.assertEqual(
            get_file_mock.call_count, len(BertPreprocessor.presets)
        )

    @unittest.mock.patch("tensorflow.keras.utils.get_file")
    def test_override_preprocessor_sequence_length(self, get_file_mock):
        get_file_mock.return_value = self.vocab
        preprocessor = BertPreprocessor.from_preset(
            "bert_base_uncased_en",
            sequence_length=64,
        )
        self.assertEqual(preprocessor.get_config()["sequence_length"], 64)
        preprocessor("The quick brown fox.")
        get_file_mock.assert_called_once()

    @unittest.mock.patch("tensorflow.keras.utils.get_file")
    def test_override_preprocessor_sequence_length_gt_max(self, get_file_mock):
        """Override sequence length longer than model's maximum."""
        get_file_mock.return_value = self.vocab
        with self.assertRaises(ValueError):
            BertPreprocessor.from_preset(
                "bert_base_uncased_en",
                sequence_length=1024,
            )
        get_file_mock.assert_called_once()
