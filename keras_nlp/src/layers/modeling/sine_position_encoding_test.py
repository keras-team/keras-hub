# Copyright 2024 The KerasNLP Authors
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

import keras
from keras import ops
from keras import random

from keras_nlp.src.layers.modeling.sine_position_encoding import (
    SinePositionEncoding,
)
from keras_nlp.src.tests.test_case import TestCase


class SinePositionEncodingTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=SinePositionEncoding,
            init_kwargs={
                "max_wavelength": 10000,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
        )

    def test_layer_behaviors_4d(self):
        self.run_layer_test(
            cls=SinePositionEncoding,
            init_kwargs={
                "max_wavelength": 10000,
            },
            input_data=random.uniform(shape=(1, 2, 4, 6)),
            expected_output_shape=(1, 2, 4, 6),
        )

    def test_static_layer_output_shape(self):
        pos_encoding = SinePositionEncoding()
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(seq_length, hidden_size))
        outputs = pos_encoding(inputs)

        # When using static positional encoding shapes, the output is expected
        # to be the same as the input shape in all dimensions.
        expected_output_shape = (None, seq_length, hidden_size)
        self.assertEqual(expected_output_shape, outputs.shape)

    def test_dynamic_layer_output_shape(self):
        pos_encoding = SinePositionEncoding()
        hidden_size = 32
        inputs = keras.Input(shape=(None, hidden_size))
        outputs = pos_encoding(inputs)

        # When using dynamic positional encoding shapes, the output is expected
        # to be the same as the input shape in all dimensions but may be None.
        expected_output_shape = (None, None, hidden_size)
        self.assertEqual(expected_output_shape, outputs.shape)

    # do multi dimension before sequence length
    def test_multi_dimension_layer_output_shape(self):
        pos_encoding = SinePositionEncoding()
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(None, seq_length, hidden_size))
        outputs = pos_encoding(inputs)

        # When using multiple dimensions before sequence length, the output is
        # expected to be the same as the input shape in all dimensions.
        expected_output_shape = (None, None, seq_length, hidden_size)
        self.assertEqual(expected_output_shape, outputs.shape)

    def test_output_correct_values(self):
        pos_encoding = SinePositionEncoding()
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                pos_encoding,
            ]
        )
        input = random.uniform(shape=[1, 4, 6])
        output = model(input)

        # comapre position encoding values for position 0 and 3
        expected_0 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        expected_3 = [0.14112, -0.98999, 0.13879, 0.99032, 0.00646, 0.99997]
        self.assertAllClose(output[0, 0, :], expected_0, atol=0.01, rtol=0.01)
        self.assertAllClose(output[0, 3, :], expected_3, atol=0.01, rtol=0.01)

    def test_start_index(self):
        batch_size, seq_length, feature_size = 2, 3, 4
        layer = SinePositionEncoding()
        data = random.uniform(shape=(batch_size, seq_length, feature_size))
        full_output = layer(data)
        sequential_output = ops.zeros((batch_size, seq_length, feature_size))
        for i in range(seq_length):
            parial_output = layer(data[:, i : i + 1, :], start_index=i)
            sequential_output = ops.slice_update(
                sequential_output, (0, i, 0), parial_output
            )
        self.assertAllClose(full_output, sequential_output)
