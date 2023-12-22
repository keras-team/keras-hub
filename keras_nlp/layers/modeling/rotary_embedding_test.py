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
from keras_nlp.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_nlp.tests.test_case import TestCase


class RotaryEmbeddingTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=RotaryEmbedding,
            init_kwargs={
                "max_wavelength": 1000,
                "scaling_factor": 2.0,
                "sequence_axis": 1,
                "feature_axis": -1,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
        )

    def test_layer_behaviors_4d(self):
        self.run_layer_test(
            cls=RotaryEmbedding,
            init_kwargs={
                "max_wavelength": 1000,
            },
            input_data=random.uniform(shape=(2, 8, 4, 6)),
            expected_output_shape=(2, 8, 4, 6),
        )

    def test_dynamic_layer_output_shape(self):
        embedding_layer = RotaryEmbedding()
        hidden_size = 32
        inputs = keras.Input(shape=(None, hidden_size))
        outputs = embedding_layer(inputs)

        # When using dynamic positional encoding shapes, the output is expected
        # to be the same as the input shape in all dimensions but may be None.
        expected_output_shape = (None, None, hidden_size)
        self.assertEqual(expected_output_shape, outputs.shape)

    # do multi dimension before sequence length
    def test_multi_dimension_layer_output_shape(self):
        embedding_layer = RotaryEmbedding()
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(None, seq_length, hidden_size))
        outputs = embedding_layer(inputs)

        # When using multiple dimensions before sequence length, the output is
        # expected to be the same as the input shape in all dimensions.
        expected_output_shape = (None, None, seq_length, hidden_size)
        self.assertEqual(expected_output_shape, outputs.shape)

    def test_output_correct_values(self):
        embedding_layer = RotaryEmbedding()
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                embedding_layer,
            ]
        )
        input = ops.ones(shape=[1, 4, 6])
        output = model(input)

        # comapre position encoding values for position 0 and 3
        expected_0 = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        expected_3 = [-1.1311, 0.8515, 0.9935, -0.8489, 1.1291, 1.0064]
        self.assertAllClose(output[0, 0, :], expected_0, atol=0.01, rtol=0.01)
        self.assertAllClose(output[0, 3, :], expected_3, atol=0.01, rtol=0.01)

    def test_start_index(self):
        batch_size, seq_length, feature_size = 2, 3, 4
        layer = RotaryEmbedding(seq_length)
        data = random.uniform(shape=(batch_size, seq_length, feature_size))
        full_output = layer(data)
        sequential_output = ops.zeros((batch_size, seq_length, feature_size))
        for i in range(seq_length):
            parial_output = layer(data[:, i : i + 1, :], start_index=i)
            sequential_output = ops.slice_update(
                sequential_output, (0, i, 0), parial_output
            )
        self.assertAllClose(full_output, sequential_output)

    def test_permuted_axes(self):
        batch_size, seq_length, feature_size = 2, 3, 4
        data = random.uniform(shape=(batch_size, seq_length, feature_size))
        layer = RotaryEmbedding(seq_length)
        outputs = layer(data)
        permuted_data = ops.transpose(data, (0, 2, 1))
        permuted_layer = RotaryEmbedding(
            seq_length, sequence_axis=-1, feature_axis=-2
        )
        permuted_outputs = permuted_layer(permuted_data)
        self.assertAllClose(outputs, ops.transpose(permuted_outputs, (0, 2, 1)))

    def test_float16_dtype(self):
        embedding_layer = RotaryEmbedding(dtype="float16")
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(seq_length, hidden_size))
        outputs = embedding_layer(inputs)

        # output dtype for this layer should be float16.
        self.assertEqual(outputs.dtype, "float16")
