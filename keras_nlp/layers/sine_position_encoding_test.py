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
"""Tests for Sinusoidal Positional encoding."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import sine_position_encoding


class SinePositionEncodingTest(tf.test.TestCase):
    def test_valid_call(self):
        pos_encoding = sine_position_encoding.SinePositionEncoding()
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                pos_encoding,
            ]
        )
        input = tf.random.uniform(shape=[2, 4, 6])
        model(input)

    def test_static_layer_output_shape(self):
        pos_encoding = sine_position_encoding.SinePositionEncoding()
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(seq_length, hidden_size))
        outputs = pos_encoding(inputs)

        # When using static positional encoding shapes, the output is expected
        # to be the same as the input shape in all dimensions.
        expected_output_shape = [None, seq_length, hidden_size]
        self.assertEqual(expected_output_shape, outputs.shape.as_list())

    def test_dynamic_layer_output_shape(self):
        pos_encoding = sine_position_encoding.SinePositionEncoding()
        hidden_size = 32
        inputs = keras.Input(shape=(None, hidden_size))
        outputs = pos_encoding(inputs)

        # When using dynamic positional encoding shapes, the output is expected
        # to be the same as the input shape in all dimensions but may be None.
        expected_output_shape = [None, None, hidden_size]
        self.assertEqual(expected_output_shape, outputs.shape.as_list())

    # do multi dimension before sequence length
    def test_multi_dimension_layer_output_shape(self):
        pos_encoding = sine_position_encoding.SinePositionEncoding()
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(None, seq_length, hidden_size))
        outputs = pos_encoding(inputs)

        # When using muliple dimensions before sequence length, the output is
        # expected to be the same as the input shape in all dimensions.
        expected_output_shape = [None, None, seq_length, hidden_size]
        self.assertEqual(expected_output_shape, outputs.shape.as_list())

    def test_output_correct_values(self):
        pos_encoding = sine_position_encoding.SinePositionEncoding()
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                pos_encoding,
            ]
        )
        input = tf.random.uniform(shape=[1, 4, 6])
        output = model(input)

        # comapre position encoding values for position 0 and 3
        expected_0 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        expected_3 = [0.14112, -0.98999, 0.13879, 0.99032, 0.00646, 0.99997]
        self.assertAllClose(output[0, 0, :], expected_0, atol=0.01, rtol=0.01)
        self.assertAllClose(output[0, 3, :], expected_3, atol=0.01, rtol=0.01)

    def test_ragged_tensor_with_3_dimensions(self):
        feature_size = 2
        test_layer = sine_position_encoding.SinePositionEncoding()
        # Create a 3-dimensional ragged input (the first dimension is implicit).
        input_tensor = keras.Input(shape=(None, feature_size), ragged=True)
        output_tensor = test_layer(input_tensor)
        model = keras.Model(input_tensor, output_tensor)

        inputs = tf.ragged.constant(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0]],
            ],
            ragged_rank=1,
            inner_shape=(2,),
        )
        expected_outputs = tf.ragged.constant(
            [
                [[0.0, 1.0], [0.84147096, 0.5403023]],
                [],
                [[0.0, 1.0], [0.84147096, 0.5403023], [0.9092974, -0.41614684]],
                [[0.0, 1.0]],
            ],
            ragged_rank=1,
            inner_shape=(2,),
        )
        outputs = model.predict(inputs)
        self.assertAllClose(outputs, expected_outputs, atol=0.01, rtol=0.01)

    def test_ragged_tensor_with_4_dimensions(self):
        feature_size = 2
        test_layer = sine_position_encoding.SinePositionEncoding()
        # Create a 4-dimensional ragged input (the first dimension is implicit).
        input_tensor = keras.Input(
            shape=(None, None, feature_size), ragged=True
        )
        output_tensor = test_layer(input_tensor)
        model = keras.Model(input_tensor, output_tensor)

        inputs = tf.ragged.constant(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [],
                ],
                [
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0]],
                ],
            ],
            ragged_rank=2,
            inner_shape=(2,),
        )
        expected_outputs = tf.ragged.constant(
            [
                [
                    [[0.0, 1.0], [0.84147096, 0.5403023]],
                    [],
                ],
                [
                    [
                        [0.0, 1.0],
                        [0.84147096, 0.5403023],
                        [0.9092974, -0.41614684],
                    ],
                    [[0.0, 1.0]],
                ],
            ],
            ragged_rank=2,
            inner_shape=(2,),
        )
        outputs = model.predict(inputs)
        self.assertAllClose(outputs, expected_outputs, atol=0.01, rtol=0.01)

    def test_get_config_and_from_config(self):
        pos_encoding = sine_position_encoding.SinePositionEncoding(
            max_wavelength=1000,
        )
        config = pos_encoding.get_config()
        expected_config_subset = {
            "max_wavelength": 1000,
        }
        self.assertEqual(config, {**config, **expected_config_subset})
        restored_pos_encoding = (
            sine_position_encoding.SinePositionEncoding.from_config(config)
        )
        self.assertEqual(
            restored_pos_encoding.get_config(),
            {**config, **expected_config_subset},
        )

    def test_float16_dtype(self):
        pos_encoding = sine_position_encoding.SinePositionEncoding(
            dtype="float16"
        )
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(seq_length, hidden_size))
        outputs = pos_encoding(inputs)

        # output dtype for this layer should be tf.float16.
        self.assertEqual(outputs.dtype, tf.float16)
