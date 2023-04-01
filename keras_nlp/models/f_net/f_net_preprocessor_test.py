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

"""Tests for FNet preprocessor layer."""

import io
import os

import pytest
import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.f_net.f_net_preprocessor import FNetPreprocessor
from keras_nlp.models.f_net.f_net_tokenizer import FNetTokenizer


class FNetPreprocessorTest(tf.test.TestCase, parameterized.TestCase):
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

        self.preprocessor = FNetPreprocessor(
            tokenizer=FNetTokenizer(proto=self.proto),
            sequence_length=12,
        )

    def test_tokenize_strings(self):
        input_data = "the quick brown fox"
        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"], [4, 2, 10, 6, 8, 5, 3, 3, 3, 3, 3, 3]
        )
        self.assertAllEqual(
            output["segment_ids"], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )

    def test_tokenize_list_of_strings(self):
        # We should handle a list of strings as batch.
        input_data = ["the quick brown fox"] * 4
        output = self.preprocessor(input_data)
        self.assertAllEqual(
            output["token_ids"],
            [[4, 2, 10, 6, 8, 5, 3, 3, 3, 3, 3, 3]] * 4,
        )
        self.assertAllEqual(
            output["segment_ids"], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 4
        )

    def test_tokenize_labeled_batch(self):
        x = tf.constant(["the quick brown fox"] * 4)
        y = tf.constant([1] * 4)
        sw = tf.constant([1.0] * 4)
        x_out, y_out, sw_out = self.preprocessor(x, y, sw)
        self.assertAllEqual(
            x_out["token_ids"],
            [[4, 2, 10, 6, 8, 5, 3, 3, 3, 3, 3, 3]] * 4,
        )
        self.assertAllEqual(
            x_out["segment_ids"], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 4
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
            [[4, 2, 10, 6, 8, 5, 3, 3, 3, 3, 3, 3]] * 4,
        )
        self.assertAllEqual(
            x_out["segment_ids"], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(y_out, y)
        self.assertAllEqual(sw_out, sw)

    def test_tokenize_multiple_sentences(self):
        sentence_one = tf.constant("the quick brown fox")
        sentence_two = tf.constant("the earth")
        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"],
            [4, 2, 10, 6, 8, 5, 2, 7, 5, 3, 3, 3],
        )
        self.assertAllEqual(
            output["segment_ids"], [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
        )

    def test_tokenize_multiple_batched_sentences(self):
        sentence_one = tf.constant(["the quick brown fox"] * 4)
        sentence_two = tf.constant(["the earth"] * 4)
        # The first tuple or list is always interpreted as an enumeration of
        # separate sequences to concatenate.
        output = self.preprocessor((sentence_one, sentence_two))
        self.assertAllEqual(
            output["token_ids"],
            [[4, 2, 10, 6, 8, 5, 2, 7, 5, 3, 3, 3]] * 4,
        )
        self.assertAllEqual(
            output["segment_ids"], [[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]] * 4
        )

    def test_errors_for_2d_list_input(self):
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            self.preprocessor(ambiguous_input)

    def test_serialization(self):
        config = keras.utils.serialize_keras_object(self.preprocessor)
        new_preprocessor = keras.utils.deserialize_keras_object(config)
        self.assertEqual(
            new_preprocessor.get_config(),
            self.preprocessor.get_config(),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        input_data = tf.constant(["the quick brown fox"])
        inputs = keras.Input(dtype="string", shape=())
        outputs = self.preprocessor(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model(input_data)["token_ids"],
            restored_model(input_data)["token_ids"],
        )
