import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma3.gemma3_layers import Gemma3InterleaveEmbeddings
from keras_hub.src.models.gemma3.gemma3_layers import Gemma3MeanPooling
from keras_hub.src.tests.test_case import TestCase


class Gemma3MeanPoolingTest(TestCase, parameterized.TestCase):
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

        layer = Gemma3MeanPooling()
        inputs = ops.convert_to_tensor(sequence_output)
        padding_mask_tensor = ops.convert_to_tensor(padding_mask)
        actual_output = layer(inputs, padding_mask=padding_mask_tensor)

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

        layer = Gemma3MeanPooling()
        inputs = ops.convert_to_tensor(sequence_output)
        padding_mask_tensor = ops.convert_to_tensor(padding_mask)
        actual_output = layer(inputs, padding_mask=padding_mask_tensor)

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

        layer = Gemma3MeanPooling(dtype=dtype)
        inputs = ops.convert_to_tensor(sequence_output)
        padding_mask_tensor = ops.convert_to_tensor(padding_mask)
        actual_output = layer(inputs, padding_mask=padding_mask_tensor)

        self.assertAllClose(actual_output, expected_output)

    def test_shape_computation(self):
        """Validates the compute_output_shape method."""
        layer = Gemma3MeanPooling()
        input_shape = [(2, 10, 16), (2, 10)]
        output_shape = layer.compute_output_shape(input_shape)
        expected_shape = (2, 16)
        self.assertEqual(output_shape, expected_shape)

    def test_config_serialization(self):
        """Tests that the layer can be successfully saved and loaded."""
        layer = Gemma3MeanPooling(name="mean_pooling_test")
        config = layer.get_config()
        new_layer = Gemma3MeanPooling.from_config(config)
        self.assertEqual(new_layer.name, layer.name)
        self.assertIsInstance(new_layer, Gemma3MeanPooling)


class Gemma3InterleaveEmbeddingsTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.max_images_per_prompt = 3
        self.num_vision_tokens_per_image = 2
        self.embedding_dim = 4
        self.seq_length = 10

        self.image_embeddings = np.random.randn(
            self.batch_size * self.max_images_per_prompt,
            self.num_vision_tokens_per_image,
            self.embedding_dim,
        ).astype("float32")

        self.text_embeddings = np.random.randn(
            self.batch_size, self.seq_length, self.embedding_dim
        ).astype("float32")

        self.vision_indices = np.array(
            [
                [1, 2, 0, 0, 0, 0],  # just one image
                [3, 4, 7, 8, 0, 0],  # two images
            ]
        )

    def test_call(self):
        reconstructed = Gemma3InterleaveEmbeddings(
            num_vision_tokens_per_image=self.num_vision_tokens_per_image,
        )(
            self.image_embeddings,
            self.text_embeddings,
            self.vision_indices,
        )

        self.assertEqual(
            ops.shape(reconstructed),
            (self.batch_size, self.seq_length, self.embedding_dim),
        )

        # First, verify `index = 0` is not touched.
        self.assertAllEqual(
            reconstructed[:, 0, :], self.text_embeddings[:, 0, :]
        )

        # Verify interleaving.
        self.assertAllEqual(
            reconstructed[0, 1:3],
            np.reshape(self.image_embeddings[0], (2, self.embedding_dim)),
        )
        self.assertAllEqual(
            reconstructed[1, 3:5],
            np.reshape(self.image_embeddings[3], (2, self.embedding_dim)),
        )
        self.assertAllEqual(
            reconstructed[1, 7:9],
            np.reshape(self.image_embeddings[4], (2, self.embedding_dim)),
        )

        # Verify everything else is unchanged.
        self.assertAllEqual(reconstructed[0, 3:], self.text_embeddings[0, 3:])
        self.assertAllEqual(reconstructed[1, 1:3], self.text_embeddings[1, 1:3])
        self.assertAllEqual(reconstructed[1, 5:7], self.text_embeddings[1, 5:7])
        self.assertAllEqual(reconstructed[1, 9:], self.text_embeddings[1, 9:])
