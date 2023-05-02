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

"""Tests for FNet tokenizer."""

import io
import os

import pytest
import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.f_net.f_net_tokenizer import FNetTokenizer


class FNetTokenizerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=12,
            model_type="WORD",
            pad_id=3,
            unk_id=0,
            bos_id=4,
            eos_id=5,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="[CLS]",
            eos_piece="[SEP]",
            user_defined_symbols="[MASK]",
        )
        self.proto = bytes_io.getvalue()

        self.tokenizer = FNetTokenizer(proto=self.proto)

    def test_tokenize(self):
        input_data = "the quick brown fox"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [2, 10, 6, 8])

    def test_tokenize_batch(self):
        input_data = tf.constant(["the quick brown fox", "the earth is round"])
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[2, 10, 6, 8], [2, 7, 9, 11]])

    def test_detokenize(self):
        input_data = tf.constant([[2, 10, 6, 8]])
        output = self.tokenizer.detokenize(input_data)
        self.assertEqual(output, tf.constant(["the quick brown fox"]))

    def test_vocabulary_size(self):
        tokenizer = FNetTokenizer(proto=self.proto)
        self.assertEqual(tokenizer.vocabulary_size(), 12)

    def test_errors_missing_special_tokens(self):
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=iter(["abc"]),
            model_writer=bytes_io,
            vocab_size=5,
            pad_id=-1,
            eos_id=-1,
            bos_id=-1,
        )
        with self.assertRaises(ValueError):
            FNetTokenizer(proto=bytes_io.getvalue())

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
        input_data = tf.constant(["the quick brown fox"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.tokenizer(inputs)
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
