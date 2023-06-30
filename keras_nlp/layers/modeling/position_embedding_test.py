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
"""Tests for position embedding layer."""

import os

from absl.testing import parameterized

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling import position_embedding
from keras_nlp.tests.test_case import TestCase


def custom_init(shape, dtype=None):
    count = 1
    for length in shape:
        count *= length
    return ops.reshape(ops.arange(count, dtype=dtype), shape)


class PositionEmbeddingTest(TestCase):
    def test_static_layer_output_shape(self):
        # Create a 3-dimensional input (the first dimension is implicit).
        sequence_length = 21
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=sequence_length
        )
        input_tensor = keras.Input(shape=(sequence_length, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using static position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions save batch.
        expected_output_shape = (None, sequence_length, feature_size)
        self.assertEqual(expected_output_shape, output_tensor.shape)
        # The output dtype for this layer should match the compute dtype.
        self.assertEqual(test_layer.compute_dtype, output_tensor.dtype)

    def test_more_than_3_dimensions_static(self):
        # Create a 4-dimensional input (the first dimension is implicit).
        sequence_length = 21
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=sequence_length
        )
        input_tensor = keras.Input(
            shape=(feature_size, sequence_length, feature_size)
        )
        output_tensor = test_layer(input_tensor)

        # When using static position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions save batch.
        expected_output_shape = (
            None,
            feature_size,
            sequence_length,
            feature_size,
        )
        self.assertEqual(expected_output_shape, output_tensor.shape)
        # The output dtype for this layer should match the compute dtype.
        self.assertEqual(test_layer.compute_dtype, output_tensor.dtype)

    def test_float16_dtype(self):
        # Create a 3-dimensional input (the first dimension is implicit).
        sequence_length = 21
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=sequence_length, dtype="float16"
        )
        input_tensor = keras.Input(shape=(sequence_length, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using static position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions save batch.
        expected_output_shape = (None, sequence_length, feature_size)
        self.assertEqual(expected_output_shape, output_tensor.shape)
        # The default output dtype for this layer should be "float32".
        self.assertEqual("float16", output_tensor.dtype)

    def test_dynamic_layer_output_shape(self):
        max_sequence_length = 40
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=max_sequence_length
        )
        # Create a 3-dimensional input (the first dimension is implicit).
        input_tensor = keras.Input(shape=(None, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using dynamic position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions - but may be None
        # if the input shape is None there.
        expected_output_shape = (None, None, feature_size)
        self.assertEqual(expected_output_shape, output_tensor.shape)

    def test_more_than_3_dimensions_dynamic(self):
        max_sequence_length = 60
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=max_sequence_length
        )
        # Create a 4-dimensional input (the first dimension is implicit).
        input_tensor = keras.Input(shape=(None, None, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using dynamic position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions - but may be None
        # if the input shape is None there.
        expected_output_shape = (None, None, None, feature_size)
        self.assertEqual(expected_output_shape, output_tensor.shape)

    def test_dynamic_layer_slicing(self):
        max_sequence_length = 40
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=max_sequence_length
        )
        # Create a 3-dimensional input (the first dimension is implicit).
        input_tensor = keras.Input(shape=(None, feature_size))
        output_tensor = test_layer(input_tensor)

        model = keras.Model(input_tensor, output_tensor)

        # Create input data that is shorter than max_sequence_length, which
        # should trigger a down-slice.
        input_length = 17
        # Note: This test explicitly uses a batch size of 1. This is to get
        # around Keras' restriction on Model invocations: inputs are expected to
        # have the same batch cardinality as outputs. In practice, this layer
        # should be used inside a model, where it can be projected when added to
        # another tensor.
        input_data = ops.ones(shape=[1, input_length, feature_size])
        output_data = model.predict(input_data)

        self.assertAllEqual([1, input_length, feature_size], output_data.shape)

    def test_callable_initializer(self):
        max_sequence_length = 4
        feature_size = 3
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=max_sequence_length,
            initializer=custom_init,
        )
        inputs = keras.Input(shape=(max_sequence_length, feature_size))
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        batch_size = 2
        data = ops.zeros(shape=[batch_size, max_sequence_length, feature_size])
        model(data)
        model_output = model.predict(data)
        expected_output = ops.broadcast_to(
            ops.reshape(
                ops.arange(max_sequence_length * feature_size),
                [max_sequence_length, feature_size],
            ),
            [batch_size, max_sequence_length, feature_size],
        )
        self.assertAllClose(model_output, expected_output)

    def test_one_training_step(self):
        max_sequence_length = 4
        feature_size = 3
        inputs = keras.Input(shape=(max_sequence_length, feature_size))
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=max_sequence_length
        )
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        batch_size = 2
        data = ops.random.uniform(
            shape=[batch_size, max_sequence_length, feature_size]
        )
        label = ops.random.uniform(
            shape=[batch_size, max_sequence_length, feature_size]
        )

        loss = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam()
        model.compile(loss=loss, optimizer=optimizer)
        loss = model.train_on_batch(x=data, y=label)
        self.assertGreater(loss, 0)

    def test_get_config_and_from_config(self):
        max_sequence_length = 40
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=max_sequence_length,
            initializer="zeros",
        )
        config = test_layer.get_config()
        restored = position_embedding.PositionEmbedding.from_config(config)
        self.assertEqual(restored.get_config(), config)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        max_sequence_length = 4
        feature_size = 6
        test_layer = position_embedding.PositionEmbedding(
            sequence_length=max_sequence_length
        )
        inputs = keras.Input(shape=(max_sequence_length, feature_size))
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        data = ops.zeros(shape=[2, max_sequence_length, feature_size])
        model(data)

        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)
        loaded_model = keras.models.load_model(path)

        model_output = model.predict(data)
        loaded_model_output = loaded_model.predict(data)
        self.assertAllClose(model_output, loaded_model_output)
