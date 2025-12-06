import numpy as np
from absl.testing import parameterized
from keras import ops
from keras.src import testing

from keras_hub.src.models.gemma3.gemma3_mean_pooling import MeanPooling


class MeanPoolingTest(testing.TestCase, parameterized.TestCase):
    def test_basic_pooling_with_padding(self):
        """Tests if the pooling correctly averages non-padded tokens."""
        sequence_output = np.array(
            [
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [99.0, 99.0, 99.0],
                ],
                [
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0],
                ],
            ],
            dtype="float32",
        )
        padding_mask = np.array(
            [
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype="int32",
        )

        expected_output = np.array(
            [
                [2.5, 3.5, 4.5],
                [13.0, 14.0, 15.0],
            ],
            dtype="float32",
        )

        layer = MeanPooling()
        inputs = [
            ops.convert_to_tensor(sequence_output),
            ops.convert_to_tensor(padding_mask),
        ]
        actual_output = layer(inputs)

        self.assertAllClose(actual_output, expected_output)

    def test_all_padded_sequence(self):
        """Tests that an all-padded sequence results in a zero vector."""
        sequence_output = np.random.rand(2, 3, 4).astype("float32")
        padding_mask = np.array(
            [
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype="int32",
        )
        expected_output_for_padded_seq = np.zeros(4, dtype="float32")

        layer = MeanPooling()
        inputs = [
            ops.convert_to_tensor(sequence_output),
            ops.convert_to_tensor(padding_mask),
        ]
        actual_output = layer(inputs)

        self.assertAllClose(actual_output[1], expected_output_for_padded_seq)

    @parameterized.named_parameters(
        ("float32", "float32"),
        ("float16", "float16"),
    )
    def test_different_dtypes(self, dtype):
        """Ensures the layer works with various float dtypes."""
        sequence_output = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=dtype)
        padding_mask = np.array([[1, 0]], dtype="int32")
        expected_output = np.array([[1.0, 2.0]], dtype=dtype)

        layer = MeanPooling(dtype=dtype)
        inputs = [
            ops.convert_to_tensor(sequence_output),
            ops.convert_to_tensor(padding_mask),
        ]
        actual_output = layer(inputs)

        self.assertAllClose(actual_output, expected_output)

    def test_shape_computation(self):
        """Validates the compute_output_shape method."""
        layer = MeanPooling()
        input_shape = [(2, 10, 16), (2, 10)]
        output_shape = layer.compute_output_shape(input_shape)
        expected_shape = (2, 16)
        self.assertEqual(output_shape, expected_shape)

    def test_config_serialization(self):
        """Tests that the layer can be successfully saved and loaded."""
        layer = MeanPooling(name="mean_pooling_test")
        config = layer.get_config()
        new_layer = MeanPooling.from_config(config)
        self.assertEqual(new_layer.name, layer.name)
        self.assertIsInstance(new_layer, MeanPooling)
