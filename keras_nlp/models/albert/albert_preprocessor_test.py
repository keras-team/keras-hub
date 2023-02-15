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

"""Tests for ALBERT preprocessor layer."""

import io
import os

import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.albert.albert_preprocessor import AlbertPreprocessor
from keras_nlp.models.albert.albert_tokenizer import AlbertTokenizer


class AlbertPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
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
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="[CLS]",
            eos_piece="[SEP]",
            user_defined_symbols="[MASK]",
        )
        self.proto = bytes_io.getvalue()

        self.preprocessor = AlbertPreprocessor(
            tokenizer=AlbertTokenizer(proto=self.proto),
            sequence_length=12,
        )

    def test_tokenize_strings(self):
        input_data = "the quick brown fox"
        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"], [2, 5, 10, 6, 8, 3, 0, 0, 0, 0, 0, 0]
        )
        self.assertAllEqual(
            output["segment_ids"], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        )

    def test_tokenize_list_of_strings(self):
        # We should handle a list of strings as as batch.
        input_data = ["the quick brown fox"] * 4
        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"],
            [[2, 5, 10, 6, 8, 3, 0, 0, 0, 0, 0, 0]] * 4,
        )
        self.assertAllEqual(
            output["segment_ids"], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] * 4
        )

    def test_tokenize_labeled_batch(self):
        x = tf.constant(["the quick brown fox"] * 4)
        y = tf.constant([1] * 4)
        sw = tf.constant([1.0] * 4)
        x_out, y_out, sw_out = self.preprocessor(x, y, sw)
        self.assertAllEqual(
            x_out["token_ids"],
            [[2, 5, 10, 6, 8, 3, 0, 0, 0, 0, 0, 0]] * 4,
        )
        self.assertAllEqual(
            x_out["segment_ids"], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            x_out["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(y_out, y)
        self.assertAllEqual(sw_out, sw)

    def test_tokenize_labeled_dataset(self):
        x = tf.constant(["the quick brown fox"] * 4)
        y = tf.constant([1] * 4)
        sw = tf.constant([1.0] * 4)
        ds = tf.data.Dataset.from_tensor_slices((x, y, sw))
        ds = ds.map(self.preprocessor)
        x_out, y_out, sw_out = ds.batch(4).take(1).get_single_element()
        self.assertAllEqual(
            x_out["token_ids"],
            [[2, 5, 10, 6, 8, 3, 0, 0, 0, 0, 0, 0]] * 4,
        )
        self.assertAllEqual(
            x_out["segment_ids"], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            x_out["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(y_out, y)
        self.assertAllEqual(sw_out, sw)

    def test_tokenize_multiple_sentences(self):
        sentence_one = tf.constant("the quick brown fox")
        sentence_two = tf.constant("the earth")
        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"],
            [2, 5, 10, 6, 8, 3, 5, 7, 3, 0, 0, 0],
        )
        self.assertAllEqual(
            output["segment_ids"], [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        )
        self.assertAllEqual(
            output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        )

    def test_tokenize_multiple_batched_sentences(self):
        sentence_one = tf.constant(["the quick brown fox"] * 4)
        sentence_two = tf.constant(["the earth"] * 4)
        # The first tuple or list is always interpreted as an enumeration of
        # separate sequences to concatenate.
        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"],
            [[2, 5, 10, 6, 8, 3, 5, 7, 3, 0, 0, 0]] * 4,
        )
        self.assertAllEqual(
            output["segment_ids"], [[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            output["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * 4
        )

    def test_errors_for_2d_list_input(self):
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            self.preprocessor(ambiguous_input)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["the quick brown fox"])
        inputs = keras.Input(dtype="string", shape=())
        outputs = self.preprocessor(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data)["token_ids"],
            restored_model(input_data)["token_ids"],
        )
