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

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.backend import random
from keras_nlp.layers.modeling.alibi_bias import AlibiBias
from keras_nlp.tests.test_case import TestCase


class AlibiBiasTest(TestCase):
    def test_layer_behaviors(self):
        alibi_bias_max = 8
        batch_size = 4
        num_heads = 8
        query_length = 10
        key_length = 10
        self.run_layer_test(
            cls=AlibiBias,
            init_kwargs={
                "alibi_bias_max": alibi_bias_max,
            },
            input_data=random.uniform(
                shape=(batch_size, num_heads, query_length, key_length)
            ),
            expected_output_shape=(
                batch_size,
                num_heads,
                query_length,
                key_length,
            ),
        )

    def test_float16_dtype(self):
        # Create a 4-dimensional input (the first dimension is implicit).
        alibi_bias_max = 8
        num_heads = 8
        query_length = 5
        key_length = 10
        test_layer = AlibiBias(alibi_bias_max=alibi_bias_max, dtype="float16")
        input_tensor = keras.Input(shape=(num_heads, query_length, key_length))
        output_tensor = test_layer(input_tensor)

        # the output is expected to be the same as the input shape in all
        # dimensions. here, the first dimension is implicit and is for batch
        expected_output_shape = (None, num_heads, query_length, key_length)
        self.assertEqual(expected_output_shape, output_tensor.shape)
        # The default output dtype for this layer should be "float32".
        self.assertEqual("float16", output_tensor.dtype)

    def test_dynamic_layer_output_shape(self):
        query_length = 10
        key_length = 10
        num_heads = 4

        test_layer = AlibiBias()
        # Create a 4-dimensional input (the first dimension is implicit).
        input_tensor = keras.Input(shape=(num_heads, query_length, key_length))
        output_tensor = test_layer(input_tensor)

        # the output is expected to be the same as the input shape in all
        # dimensions.
        expected_output_shape = (
            None,
            num_heads,
            query_length,
            key_length,
        )
        self.assertEqual(expected_output_shape, output_tensor.shape)

    def test_value_error_when_inputs_shape_is_not_4(self):
        with self.assertRaises(ValueError):
            AlibiBias()(random.uniform(shape=(12, 12)))

    def test_num_heads_is_not_power_of_two(self):
        inputs_shape = (1, 12, 12, 12)
        inputs = random.uniform(shape=inputs_shape)
        layer = AlibiBias()
        outputs = layer(inputs)
        self.assertEqual(inputs_shape, outputs.shape)

    def test_correct_output(self):
        batch_size = 1
        num_heads = 8
        query_length = 1
        key_length = 3
        input_shape = (batch_size, num_heads, query_length, key_length)
        input_tensor = ops.zeros(input_shape)
        layer = AlibiBias()
        output_tensor = layer(input_tensor)
        print(output_tensor)
        self.assertAllClose(
            output_tensor,
            ops.convert_to_tensor(
                [
                    [
                        [[-1.0, -0.5, 0.0]],
                        [[-0.5, -0.25, 0.0]],
                        [[-0.25, -0.125, 0.0]],
                        [[-0.125, -0.0625, 0.0]],
                        [[-0.0625, -0.03125, 0.0]],
                        [[-0.03125, -0.015625, 0.0]],
                        [[-0.015625, -0.0078125, 0.0]],
                        [[-0.0078125, -0.00390625, 0.0]],
                    ]
                ]
            ),
        )

    def test_correct_output_num_heads_not_power_of_two(self):
        batch_size = 1
        num_heads = 14
        query_length = 1
        key_length = 3
        input_shape = (batch_size, num_heads, query_length, key_length)
        input_tensor = ops.zeros(input_shape)
        layer = AlibiBias()
        output_tensor = layer(input_tensor)
        print(output_tensor)
        self.assertAllClose(
            output_tensor,
            ops.convert_to_tensor(
                [
                    [
                        [[-1.0, -0.5, 0.0]],
                        [[-0.5, -0.25, 0.0]],
                        [[-0.25, -0.125, 0.0]],
                        [[-0.125, -0.0625, 0.0]],
                        [[-0.0625, -0.03125, 0.0]],
                        [[-0.03125, -0.015625, 0.0]],
                        [[-0.015625, -0.0078125, 0.0]],
                        [[-0.0078125, -0.00390625, 0.0]],
                        [[-1.4142135, -0.70710677, 0.0]],
                        [[-0.70710677, -0.35355338, 0.0]],
                        [[-0.35355338, -0.17677669, 0.0]],
                        [[-0.17677669, -0.08838835, 0.0]],
                        [[-0.08838835, -0.04419417, 0.0]],
                        [[-0.04419417, -0.02209709, 0.0]],
                    ]
                ]
            ),
        )

    def test_correct_output_alibi_bias_max(self):
        alibi_bias_max = 12
        batch_size = 1
        num_heads = 2
        query_length = 1
        key_length = 3
        input_shape = (batch_size, num_heads, query_length, key_length)
        input_tensor = ops.zeros(input_shape)
        layer = AlibiBias(alibi_bias_max=alibi_bias_max)
        output_tensor = layer(input_tensor)
        print(output_tensor)
        self.assertAllClose(
            output_tensor,
            ops.convert_to_tensor(
                [
                    [
                        [[-0.03125, -0.015625, 0.0]],
                        [[-0.00048828, -0.00024414, 0.0]],
                    ]
                ]
            ),
        )

    def test_correct_output_alibi_bias_max_num_heads_not_power_of_two(
        self,
    ):
        alibi_bias_max = 6
        batch_size = 1
        num_heads = 3
        query_length = 1
        key_length = 3
        input_shape = (batch_size, num_heads, query_length, key_length)
        input_tensor = ops.zeros(input_shape)
        layer = AlibiBias(alibi_bias_max=alibi_bias_max)
        output_tensor = layer(input_tensor)
        print(output_tensor)
        self.assertAllClose(
            output_tensor,
            ops.convert_to_tensor(
                [
                    [
                        [[-0.25, -0.125, 0.0]],
                        [[-0.03125, -0.015625, 0.0]],
                        [[-0.70710677, -0.35355338, 0.0]],
                    ]
                ]
            ),
        )
