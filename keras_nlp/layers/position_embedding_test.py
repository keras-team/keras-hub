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
"""Tests for position embedding layer."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import position_embedding


def custom_init(shape, dtype=None):
    count = 1
    for length in shape:
        count *= length
    return tf.reshape(tf.range(count, dtype=dtype), shape)


class PositionEmbeddingLayerTest(tf.test.TestCase):
    def test_static_layer_output_shape(self):
        # Create a 3-dimensional input (the first dimension is implicit).
        sequence_length = 21
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            max_length=sequence_length
        )
        input_tensor = tf.keras.Input(shape=(sequence_length, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using static position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions save batch.
        expected_output_shape = [None, sequence_length, feature_size]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        # The default output dtype for this layer should be tf.float32.
        self.assertEqual(tf.float32, output_tensor.dtype)

    def test_more_than_3_dimensions_static(self):
        # Create a 4-dimensional input (the first dimension is implicit).
        sequence_length = 21
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            max_length=sequence_length
        )
        input_tensor = tf.keras.Input(
            shape=(feature_size, sequence_length, feature_size)
        )
        output_tensor = test_layer(input_tensor)

        # When using static position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions save batch.
        expected_output_shape = [
            None,
            feature_size,
            sequence_length,
            feature_size,
        ]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        # The default output dtype for this layer should be tf.float32.
        self.assertEqual(tf.float32, output_tensor.dtype)

    def test_float16_dtype(self):
        # Create a 3-dimensional input (the first dimension is implicit).
        sequence_length = 21
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            max_length=sequence_length, dtype="float16"
        )
        input_tensor = tf.keras.Input(shape=(sequence_length, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using static position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions save batch.
        expected_output_shape = [None, sequence_length, feature_size]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        # The default output dtype for this layer should be tf.float32.
        self.assertEqual(tf.float16, output_tensor.dtype)

    def test_dynamic_layer_output_shape(self):
        max_sequence_length = 40
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length
        )
        # Create a 3-dimensional input (the first dimension is implicit).
        input_tensor = tf.keras.Input(shape=(None, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using dynamic position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions - but may be None
        # if the input shape is None there.
        expected_output_shape = [None, None, feature_size]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())

    def test_more_than_3_dimensions_dynamic(self):
        max_sequence_length = 60
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length
        )
        # Create a 4-dimensional input (the first dimension is implicit).
        input_tensor = tf.keras.Input(shape=(None, None, feature_size))
        output_tensor = test_layer(input_tensor)

        # When using dynamic position embedding shapes, the output is expected
        # to be the same as the input shape in all dimensions - but may be None
        # if the input shape is None there.
        expected_output_shape = [None, None, None, feature_size]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())

    def test_dynamic_layer_slicing(self):
        max_sequence_length = 40
        feature_size = 30
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length
        )
        # Create a 3-dimensional input (the first dimension is implicit).
        input_tensor = tf.keras.Input(shape=(None, feature_size))
        output_tensor = test_layer(input_tensor)

        model = tf.keras.Model(input_tensor, output_tensor)

        # Create input data that is shorter than max_sequence_length, which
        # should trigger a down-slice.
        input_length = 17
        # Note: This test explicitly uses a batch size of 1. This is to get
        # around Keras' restriction on Model invocations: inputs are expected to
        # have the same batch cardinality as outputs. In practice, this layer
        # should be used inside a model, where it can be projected when added to
        # another tensor.
        input_data = tf.ones(shape=[1, input_length, feature_size])
        output_data = model.predict(input_data)

        self.assertAllEqual([1, input_length, feature_size], output_data.shape)

    def test_callable_initializer(self):
        max_sequence_length = 4
        feature_size = 3
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length,
            initializer=custom_init,
        )
        inputs = tf.keras.Input(shape=(max_sequence_length, feature_size))
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        batch_size = 2
        data = tf.zeros(shape=[batch_size, max_sequence_length, feature_size])
        model(data)
        model_output = model.predict(data)
        expected_output = tf.broadcast_to(
            tf.reshape(
                tf.range(max_sequence_length * feature_size),
                [max_sequence_length, feature_size],
            ),
            [batch_size, max_sequence_length, feature_size],
        )
        self.assertAllClose(model_output, expected_output)

    def test_ragged_tensor_with_3_dimensions(self):
        max_sequence_length = 4
        feature_size = 2
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length,
            initializer=custom_init,
        )
        # Create a 3-dimensional ragged input (the first dimension is implicit).
        input_tensor = tf.keras.Input(
            shape=(None, feature_size), dtype=tf.float32, ragged=True
        )
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)

        input_data = tf.ragged.constant(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0]],
            ],
            ragged_rank=1,
            inner_shape=(2,),
        )
        expected_output_data = tf.ragged.constant(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [],
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                [[0.0, 1.0]],
            ],
            ragged_rank=1,
            inner_shape=(2,),
        )
        output_data = model.predict(input_data)
        self.assertAllClose(output_data, expected_output_data)

    def test_ragged_tensor_with_4_dimensions(self):
        max_sequence_length = 4
        feature_size = 2
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length,
            initializer=custom_init,
        )
        # Create a 4-dimensional ragged input (the first dimension is implicit).
        input_tensor = tf.keras.Input(
            shape=(None, None, feature_size), dtype=tf.float32, ragged=True
        )
        output_tensor = test_layer(input_tensor)
        model = tf.keras.Model(input_tensor, output_tensor)

        input_data = tf.ragged.constant(
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
        expected_output_data = tf.ragged.constant(
            [
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [],
                ],
                [
                    [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                    [[0.0, 1.0]],
                ],
            ],
            ragged_rank=2,
            inner_shape=(2,),
        )
        output_data = model.predict(input_data)
        self.assertAllClose(output_data, expected_output_data)

    def test_one_training_step(self):
        max_sequence_length = 4
        feature_size = 3
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length
        )
        inputs = tf.keras.Input(shape=(max_sequence_length, feature_size))
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        batch_size = 2
        data = tf.random.uniform(
            shape=[batch_size, max_sequence_length, feature_size]
        )
        label = tf.random.uniform(shape=[max_sequence_length, feature_size])

        loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            pred = model(data)
            loss = loss_fn(label, pred)
        grad = tape.gradient(loss, model.trainable_variables)
        self.assertEquals(len(grad), 1)

        trainable_variables_before = tf.Variable(model.trainable_variables[0])
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        self.assertNotAllClose(
            trainable_variables_before, model.trainable_variables[0]
        )

    def test_get_config_and_from_config(self):
        max_sequence_length = 40
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length
        )
        config = test_layer.get_config()
        expected_config_subset = {
            "max_length": max_sequence_length,
            "initializer": {
                "class_name": "GlorotUniform",
                "config": {"seed": None},
            },
        }
        self.assertEqual(config, {**config, **expected_config_subset})

        restored_encoder = position_embedding.PositionEmbedding.from_config(
            config,
        )
        self.assertEqual(
            restored_encoder.get_config(), {**config, **expected_config_subset}
        )

    def test_save_model(self):
        max_sequence_length = 4
        feature_size = 6
        test_layer = position_embedding.PositionEmbedding(
            max_length=max_sequence_length
        )
        inputs = tf.keras.Input(shape=(max_sequence_length, feature_size))
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        data = tf.zeros(shape=[2, max_sequence_length, feature_size])
        model(data)

        path = os.path.join(self.get_temp_dir(), "model")
        model.save(path)
        loaded_model = keras.models.load_model(path)

        model_output = model.predict(data)
        loaded_model_output = loaded_model.predict(data)
        self.assertAllClose(model_output, loaded_model_output)


if __name__ == "__main__":
    tf.test.main()
