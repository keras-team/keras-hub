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
"""Tests for DistilBERT tokenizer."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.distil_bert.distil_bert_tokenizer import (
    DistilBertTokenizer,
)


class DistilBertTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.tokenizer = DistilBertTokenizer(vocabulary=self.vocab)

    def test_tokenize(self):
        input_data = "THE QUICK BROWN FOX."
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [5, 6, 7, 8, 1])

    def test_tokenize_batch(self):
        input_data = tf.constant(["THE QUICK BROWN FOX.", "THE FOX."])
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[5, 6, 7, 8, 1], [5, 8, 1]])

    def test_lowercase(self):
        input_data = "THE QUICK BROWN FOX."
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab, lowercase=True)
        output = tokenizer(input_data)
        self.assertAllEqual(output, [9, 10, 11, 12, 1])

    def test_detokenize(self):
        input_tokens = [[5, 6, 7, 8]]
        output = self.tokenizer.detokenize(input_tokens)
        self.assertAllEqual(output, ["THE QUICK BROWN FOX"])

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 13)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            DistilBertTokenizer(vocabulary=["a", "b", "c"])

    def test_serialization(self):
        config = keras.utils.serialize_keras_object(self.tokenizer)
        new_tokenizer = keras.utils.deserialize_keras_object(config)
        self.assertEqual(
            new_tokenizer.get_config(),
            self.tokenizer.get_config(),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["THE QUICK BROWN FOX."])
        tokenizer = DistilBertTokenizer(vocabulary=self.vocab)
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
