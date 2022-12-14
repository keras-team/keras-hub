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
"""Tests for Transformer Decoder."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker


class MultiSegmentPackerTest(tf.test.TestCase, parameterized.TestCase):
    def test_trim_single_input_ints(self):
        input_data = tf.range(3, 10)
        packer = MultiSegmentPacker(8, start_value=1, end_value=2)
        output = packer(input_data)
        self.assertAllEqual(
            output, ([1, 3, 4, 5, 6, 7, 8, 2], [0, 0, 0, 0, 0, 0, 0, 0])
        )

    def test_trim_single_input_strings(self):
        input_data = tf.constant(["a", "b", "c", "d"])
        packer = MultiSegmentPacker(5, start_value="[CLS]", end_value="[SEP]")
        output = packer(input_data)
        self.assertAllEqual(
            output, (["[CLS]", "a", "b", "c", "[SEP]"], [0, 0, 0, 0, 0])
        )

    def test_trim_multiple_inputs_round_robin(self):
        seq1 = tf.constant(["a", "b", "c"])
        seq2 = tf.constant(["x", "y", "z"])
        packer = MultiSegmentPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncate="round_robin"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output,
            (
                ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
                [0, 0, 0, 0, 1, 1, 1],
            ),
        )

    def test_trim_multiple_inputs_waterfall(self):
        seq1 = tf.constant(["a", "b", "c"])
        seq2 = tf.constant(["x", "y", "z"])
        packer = MultiSegmentPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncate="waterfall"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output,
            (
                ["[CLS]", "a", "b", "c", "[SEP]", "x", "[SEP]"],
                [0, 0, 0, 0, 0, 1, 1],
            ),
        )

    def test_trim_batched_inputs_round_robin(self):
        seq1 = tf.constant([["a", "b", "c"], ["a", "b", "c"]])
        seq2 = tf.constant([["x", "y", "z"], ["x", "y", "z"]])
        packer = MultiSegmentPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncate="round_robin"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output,
            (
                [
                    ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
                    ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
                ],
                [
                    [0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1],
                ],
            ),
        )

    def test_trim_batched_inputs_waterfall(self):
        seq1 = tf.ragged.constant([["a", "b", "c"], ["a", "b"]])
        seq2 = tf.constant([["x", "y", "z"], ["x", "y", "z"]])
        packer = MultiSegmentPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncate="waterfall"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output,
            (
                [
                    ["[CLS]", "a", "b", "c", "[SEP]", "x", "[SEP]"],
                    ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
                ],
                [
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1],
                ],
            ),
        )

    def test_pad_inputs(self):
        seq1 = tf.constant(["a"])
        seq2 = tf.constant(["x"])
        packer = MultiSegmentPacker(
            6, start_value="[CLS]", end_value="[SEP]", pad_value="[PAD]"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output,
            (
                ["[CLS]", "a", "[SEP]", "x", "[SEP]", "[PAD]"],
                [0, 0, 0, 1, 1, 0],
            ),
        )

    def test_pad_batched_inputs(self):
        seq1 = tf.ragged.constant([["a"], ["a"]])
        seq2 = tf.ragged.constant([["x"], ["x", "y"]])
        packer = MultiSegmentPacker(
            7, start_value="[CLS]", end_value="[SEP]", pad_value="[PAD]"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output,
            (
                [
                    ["[CLS]", "a", "[SEP]", "x", "[SEP]", "[PAD]", "[PAD]"],
                    ["[CLS]", "a", "[SEP]", "x", "y", "[SEP]", "[PAD]"],
                ],
                [
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0],
                ],
            ),
        )

    def test_config(self):
        seq1 = tf.ragged.constant([["a", "b", "c"], ["a", "b"]])
        seq2 = tf.ragged.constant([["x", "y", "z"], ["x", "y", "z"]])
        original_packer = MultiSegmentPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncate="waterfall"
        )
        cloned_packer = MultiSegmentPacker.from_config(
            original_packer.get_config()
        )
        self.assertAllEqual(
            original_packer([seq1, seq2]),
            cloned_packer([seq1, seq2]),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        seq1 = tf.ragged.constant([["a", "b", "c"], ["a", "b"]])
        seq2 = tf.ragged.constant([["x", "y", "z"], ["x", "y", "z"]])
        packer = MultiSegmentPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncate="waterfall"
        )
        inputs = (
            keras.Input(dtype="string", ragged=True, shape=(None,)),
            keras.Input(dtype="string", ragged=True, shape=(None,)),
        )
        outputs = packer(inputs)
        model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)
        restored_model = keras.models.load_model(path)
        self.assertAllEqual(
            model((seq1, seq2)),
            restored_model((seq1, seq2)),
        )
