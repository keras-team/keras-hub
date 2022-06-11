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

"""Tests for Start End Packer layer."""


import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.start_end_packer import StartEndPacker


class StartEndPackerTest(tf.test.TestCase):
    def test_dense_input(self):
        input_data = tf.constant([5, 6, 7])
        start_end_packer = StartEndPacker(sequence_length=5)
        output = start_end_packer(input_data)
        expected_output = [5, 6, 7, 0, 0]
        self.assertAllEqual(output, expected_output)

    def test_dense_input_error(self):
        input_data = tf.constant([[5, 6, 7]])
        start_end_packer = StartEndPacker(sequence_length=5)
        with self.assertRaises(ValueError):
            start_end_packer(input_data)

    def test_ragged_input(self):
        input_data = tf.ragged.constant([[5, 6, 7], [8, 9, 10, 11]])
        start_end_packer = StartEndPacker(sequence_length=5)
        output = start_end_packer(input_data)
        expected_output = [[5, 6, 7, 0, 0], [8, 9, 10, 11, 0]]
        self.assertAllEqual(output, expected_output)

    def test_ragged_input_error(self):
        input_data = tf.ragged.constant([[[5, 6, 7], [8, 9, 10, 11]]])
        start_end_packer = StartEndPacker(sequence_length=5)
        with self.assertRaises(ValueError):
            start_end_packer(input_data)

    def test_start_end_token(self):
        input_data = tf.ragged.constant([[5, 6, 7], [8, 9, 10, 11]])
        start_end_packer = StartEndPacker(
            sequence_length=6, start_value=1, end_value=2
        )
        output = start_end_packer(input_data)
        expected_output = [[1, 5, 6, 7, 2, 0], [1, 8, 9, 10, 11, 2]]
        self.assertAllEqual(output, expected_output)

    def test_start_end_padding_value(self):
        input_data = tf.ragged.constant([[5, 6, 7], [8, 9, 10, 11]])
        start_end_packer = StartEndPacker(
            sequence_length=7, start_value=1, end_value=2, pad_value=3
        )
        output = start_end_packer(input_data)
        expected_output = [[1, 5, 6, 7, 2, 3, 3], [1, 8, 9, 10, 11, 2, 3]]
        self.assertAllEqual(output, expected_output)

    def test_string_input(self):
        input_data = tf.ragged.constant(
            [["KerasNLP", "is", "awesome"], ["amazing"]]
        )
        start_end_packer = StartEndPacker(
            sequence_length=5,
            start_value="[START]",
            end_value="[END]",
            pad_value="[PAD]",
        )
        output = start_end_packer(input_data)
        expected_output = [
            ["[START]", "KerasNLP", "is", "awesome", "[END]"],
            ["[START]", "amazing", "[END]", "[PAD]", "[PAD]"],
        ]
        self.assertAllEqual(output, expected_output)

    def test_functional_model(self):
        input_data = tf.ragged.constant([[5, 6, 7], [8, 9, 10, 11]])
        start_end_packer = StartEndPacker(
            sequence_length=7, start_value=1, end_value=2, pad_value=3
        )

        inputs = keras.Input(dtype=tf.int32, shape=())
        outputs = start_end_packer(inputs)
        model = keras.Model(inputs, outputs)
        model_output = model(input_data)

        expected_output = [[1, 5, 6, 7, 2, 3, 3], [1, 8, 9, 10, 11, 2, 3]]
        self.assertAllEqual(model_output, expected_output)
