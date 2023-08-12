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

"""Tests for XLNET tokenizer."""

import io

import sentencepiece
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.models.xlnet.xlnet_tokenizer import XLNetTokenizer
from keras_nlp.tests.test_case import TestCase


class XLNetTokenizerTest(TestCase):
    def setUp(self):
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=14,
            model_type="WORD",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_piece="<pad>",
            bos_piece="<s>",
            eos_piece="</s>",
            unk_piece="<unk>",
            user_defined_symbols=["<mask>", "<cls>", "<sep>"],
        )
        self.proto = bytes_io.getvalue()

        self.tokenizer = XLNetTokenizer(proto=self.proto)

    def test_tokenize(self):
        input_data = ["the quick brown fox"]
        output = self.tokenizer(input_data)
        self.assertAllEqual(output, [[7, 12, 8, 10, 6, 5]])

    def test_tokenize_batch(self):
        input_data = ["the quick brown fox", "the earth is round"]
        output = self.tokenizer(input_data)
        self.assertAllEqual(
            output, [[7, 12, 8, 10, 6, 5], [7, 9, 11, 13, 6, 5]]
        )

    def test_tokenize_ds(self):
        input_ds = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )
        input_ds = input_ds.map(self.tokenizer)
        outputs = []
        for each_item in input_ds.take(2):
            self.assertTrue(isinstance(each_item, tf.RaggedTensor))
            outputs.append(each_item.to_tensor())

        outputs = tf.squeeze(tf.convert_to_tensor(outputs), 1)
        self.assertAllEqual(
            outputs,
            tf.convert_to_tensor([[7, 12, 8, 10, 6, 5], [7, 9, 11, 13, 6, 5]]),
        )

    def test_detokenize(self):
        input_data = [[7, 12, 8, 10, 6, 5]]
        output = self.tokenizer.detokenize(input_data)
        self.assertEqual(output, ["the quick brown fox"])

    def test_detokenize_mask_token(self):
        input_data = [[7, 12, 8, 10, 6, 5, self.tokenizer.mask_token_id]]
        output = self.tokenizer.detokenize(input_data)
        self.assertEqual(output, ["the quick brown fox"])

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 14)

    def test_get_vocabulary_mask_token(self):
        self.assertEqual(self.tokenizer.get_vocabulary()[4], "<mask>")

    def test_id_to_token_mask_token(self):
        self.assertEqual(self.tokenizer.id_to_token(4), "<mask>")

    def test_token_to_id_mask_token(self):
        self.assertEqual(self.tokenizer.token_to_id("<mask>"), 4)

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
            XLNetTokenizer(proto=bytes_io.getvalue())

    def test_serialization(self):
        config = keras.saving.serialize_keras_object(self.tokenizer)
        new_tokenizer = keras.saving.deserialize_keras_object(config)
        self.assertEqual(
            new_tokenizer.get_config(),
            self.tokenizer.get_config(),
        )
