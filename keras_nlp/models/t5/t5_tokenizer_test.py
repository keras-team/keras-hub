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

"""Tests for T5 tokenizer."""
import io
import os

import pytest
import sentencepiece
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.models.t5.t5_tokenizer import T5Tokenizer
from keras_nlp.tests.test_case import TestCase


class T5TokenizerTest(TestCase):
    def setUp(self):
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=11,
            model_type="WORD",
            bos_id=-1,
            pad_id=0,
            eos_id=1,
            unk_id=2,
            pad_piece="<pad>",
            eos_piece="</s>",
            unk_piece="<unk>",
            user_defined_symbols="[MASK]",
        )
        self.proto = bytes_io.getvalue()

        self.tokenizer = T5Tokenizer(proto=self.proto)

    def test_tokenize(self):
        input_data = "the quick brown fox"
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [4, 9, 5, 7])

    def test_tokenize_batch(self):
        input_data = ["the quick brown fox", "the earth is round"]
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[4, 9, 5, 7], [4, 6, 8, 10]])

    def test_detokenize(self):
        input_data = [[4, 9, 5, 7]]
        output = self.tokenizer.detokenize(input_data)
        self.assertEqual(output, ["the quick brown fox"])

    def test_vocabulary_size(self):
        tokenizer = T5Tokenizer(proto=self.proto)
        self.assertEqual(tokenizer.vocabulary_size(), 11)

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
            T5Tokenizer(proto=bytes_io.getvalue())

    def test_serialization(self):
        config = keras.saving.serialize_keras_object(self.tokenizer)
        new_tokenizer = keras.saving.deserialize_keras_object(config)
        self.assertEqual(
            new_tokenizer.get_config(),
            self.tokenizer.get_config(),
        )

    @pytest.mark.large  # Saving is slow, so mark these large.
    @pytest.mark.tf_only
    def test_saved_model(self):
        input_data = tf.constant(["the quick brown fox"])

        inputs = keras.Input(dtype="string", shape=())
        outputs = self.tokenizer(inputs)
        model = keras.Model(inputs, outputs)

        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")

        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data),
            restored_model(input_data),
        )
