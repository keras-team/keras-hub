import keras
import numpy as np
from keras import ops
from keras import random

from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.tests.test_case import TestCase


def custom_init(shape, dtype=None):
    count = 1
    for length in shape:
        count *= length
    return ops.reshape(ops.arange(count, dtype=dtype), shape)


class PositionEmbeddingTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=PositionEmbedding,
            init_kwargs={
                "sequence_length": 21,
            },
            input_data=random.uniform(shape=(4, 21, 30)),
            expected_output_shape=(4, 21, 30),
            expected_num_trainable_weights=1,
        )

    def test_layer_behaviors_4d(self):
        self.run_layer_test(
            cls=PositionEmbedding,
            init_kwargs={
                "sequence_length": 21,
            },
            input_data=random.uniform(shape=(4, 5, 21, 30)),
            expected_output_shape=(4, 5, 21, 30),
            expected_num_trainable_weights=1,
        )

    def test_float16_dtype(self):
        # Create a 3-dimensional input (the first dimension is implicit).
        sequence_length = 21
        feature_size = 30
        test_layer = PositionEmbedding(
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
        test_layer = PositionEmbedding(sequence_length=max_sequence_length)
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
        test_layer = PositionEmbedding(sequence_length=max_sequence_length)
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
        test_layer = PositionEmbedding(sequence_length=max_sequence_length)
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
        input_data = np.ones(shape=[1, input_length, feature_size])
        output_data = model.predict(input_data)

        self.assertAllEqual([1, input_length, feature_size], output_data.shape)

    def test_callable_initializer(self):
        max_sequence_length = 4
        feature_size = 3
        test_layer = PositionEmbedding(
            sequence_length=max_sequence_length,
            initializer=custom_init,
        )
        inputs = keras.Input(shape=(max_sequence_length, feature_size))
        outputs = test_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        batch_size = 2
        data = np.zeros(shape=[batch_size, max_sequence_length, feature_size])
        model(data)
        model_output = model.predict(data)
        expected_output = np.broadcast_to(
            np.reshape(
                np.arange(max_sequence_length * feature_size),
                [max_sequence_length, feature_size],
            ),
            [batch_size, max_sequence_length, feature_size],
        )
        self.assertAllClose(model_output, expected_output)

    def test_start_index(self):
        batch_size, seq_length, feature_size = 2, 3, 4
        layer = PositionEmbedding(seq_length)
        data = random.uniform(shape=(batch_size, seq_length, feature_size))
        full_output = layer(data)
        sequential_output = ops.zeros((batch_size, seq_length, feature_size))
        for i in range(seq_length):
            parial_output = layer(data[:, i : i + 1, :], start_index=i)
            sequential_output = ops.slice_update(
                sequential_output, (0, i, 0), parial_output
            )
        self.assertAllClose(full_output, sequential_output)

    def test_positions(self):
        batch_size, seq_length, feature_size = 2, 4, 5
        data = random.uniform(shape=(batch_size, seq_length, feature_size))
        positions = np.array([[0, 0, 1, 2], [1, 2, 3, 0]])

        layer = PositionEmbedding(seq_length)
        output = layer(data, positions=positions)

        expected_output = []
        for b_idx in range(batch_size):
            for s_idx in range(seq_length):
                actual_position = positions[b_idx, s_idx]
                expected_output.append(
                    layer.position_embeddings.numpy()[actual_position]
                )

        expected_output = np.reshape(
            np.array(expected_output), (batch_size, seq_length, feature_size)
        )
        self.assertAllClose(output, expected_output, rtol=1e-5, atol=1e-5)
