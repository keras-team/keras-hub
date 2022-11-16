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

import os
import unittest

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.distilbert.distilbert_preprocessing import (
    DistilBertPreprocessor,
)
from keras_nlp.models.distilbert.distilbert_preprocessing import (
    DistilBertTokenizer,
)


class DistilBertTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]

    def test_tokenize(self):
        input_data = "THE QUICK BROWN FOX."
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab)
        output = tokenizer(input_data)
        self.assertAllEqual(output, [5, 6, 7, 8, 1])

    def test_tokenize_batch(self):
        input_data = tf.constant(["THE QUICK BROWN FOX.", "THE FOX."])
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab)
        output = tokenizer(input_data)
        self.assertAllEqual(output, [[5, 6, 7, 8, 1], [5, 8, 1]])

    def test_lowercase(self):
        input_data = "THE QUICK BROWN FOX."
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab, lowercase=True)
        output = tokenizer(input_data)
        self.assertAllEqual(output, [9, 10, 11, 12, 1])

    def test_detokenize(self):
        input_tokens = [[5, 6, 7, 8]]
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab)
        output = tokenizer.detokenize(input_tokens)
        self.assertAllEqual(output, ["THE QUICK BROWN FOX"])

    def test_vocabulary_size(self):
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab)
        self.assertEqual(tokenizer.vocabulary_size(), 13)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            DistilBertTokenizer.from_preset("distilbert_base_uncased_clowntown")

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in DistilBertPreprocessor.presets:
            self.assertRegex(DistilBertPreprocessor.from_preset.__doc__, name)

    @unittest.mock.patch("tensorflow.keras.utils.get_file")
    def test_valid_call_presets(self, get_file_mock):
        """Ensure presets have necessary structure, but no RPCs."""
        input_data = ["THE QUICK BROWN FOX."]
        get_file_mock.return_value = self.vocab
        for preset in DistilBertTokenizer.presets:
            tokenizer = DistilBertTokenizer.from_preset(preset)
            tokenizer(input_data)
        self.assertEqual(
            get_file_mock.call_count, len(DistilBertTokenizer.presets)
        )

    @parameterized.named_parameters(
        ("save_format_tf", "tf"), ("save_format_h5", "h5")
    )
    def test_saving_model(self, save_format):
        input_data = tf.constant(["THE QUICK BROWN FOX."])
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab)
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), "model")
        model.save(path, save_format=save_format)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )


class DistilBertPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]

    def test_tokenize(self):
        input_data = ["THE QUICK BROWN FOX."]
        preprocessor = DistilBertPreprocessor(
            DistilBertTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        output = preprocessor(input_data)
        self.assertAllEqual(output["token_ids"], [2, 5, 6, 7, 8, 1, 3, 0])
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
        preprocessor = DistilBertPreprocessor(
            DistilBertTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        output = preprocessor(input_data)
        self.assertAllEqual(output["token_ids"], [[2, 5, 6, 7, 8, 1, 3, 0]] * 4)
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 0]] * 4
        )

    def test_tokenize_multiple_sentences(self):
        sentence_one = "THE QUICK"
        sentence_two = "BROWN FOX."
        preprocessor = DistilBertPreprocessor(
            DistilBertTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        # The first tuple or list is always interpreted as an enumeration of
        # separate sequences to concatenate.
        output = preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(output["token_ids"], [2, 5, 6, 3, 7, 8, 1, 3])
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
        preprocessor = DistilBertPreprocessor(
            DistilBertTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        # The first tuple or list is always interpreted as an enumeration of
        # separate sequences to concatenate.
        output = preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(output["token_ids"], [[2, 5, 6, 3, 7, 8, 1, 3]] * 4)
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1]] * 4
        )

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            DistilBertPreprocessor.from_preset(
                "distilbert_base_uncased_clowntown"
            )

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in DistilBertPreprocessor.presets:
            self.assertRegex(DistilBertPreprocessor.from_preset.__doc__, name)

    @unittest.mock.patch("tensorflow.keras.utils.get_file")
    def test_valid_call_presets(self, get_file_mock):
        """Ensure presets have necessary structure, but no RPCs."""
        input_data = ["THE QUICK BROWN FOX."]
        get_file_mock.return_value = self.vocab
        for preset in DistilBertPreprocessor.presets:
            preprocessor = DistilBertPreprocessor.from_preset(preset)
            preprocessor(input_data)
        self.assertEqual(
            get_file_mock.call_count, len(DistilBertPreprocessor.presets)
        )

    @parameterized.named_parameters(
        ("save_format_tf", "tf"), ("save_format_h5", "h5")
    )
    def test_saving_model(self, save_format):
        input_data = tf.constant(["THE QUICK BROWN FOX."])
        preprocessor = DistilBertPreprocessor(
            DistilBertTokenizer(vocabulary=self.vocab),
            sequence_length=8,
        )
        inputs = keras.Input(dtype="string", shape=())
        outputs = preprocessor(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), "model")
        model.save(path, save_format=save_format)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data)["token_ids"],
            restored_model(input_data)["token_ids"],
        )
