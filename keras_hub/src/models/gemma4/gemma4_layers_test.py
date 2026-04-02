import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma4.gemma4_layers import Gemma4InterleaveEmbeddings
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4MeanPooling
from keras_hub.src.tests.test_case import TestCase


class Gemma4MeanPoolingTest(TestCase, parameterized.TestCase):
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

        layer = Gemma4MeanPooling()
        inputs = ops.convert_to_tensor(sequence_output)
        padding_mask_tensor = ops.convert_to_tensor(padding_mask)
        actual_output = ops.convert_to_numpy(
            layer(inputs, padding_mask=padding_mask_tensor)
        )

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

        layer = Gemma4MeanPooling()
        inputs = ops.convert_to_tensor(sequence_output)
        padding_mask_tensor = ops.convert_to_tensor(padding_mask)
        actual_output = ops.convert_to_numpy(
            layer(inputs, padding_mask=padding_mask_tensor)
        )

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

        layer = Gemma4MeanPooling(dtype=dtype)
        inputs = ops.convert_to_tensor(sequence_output)
        padding_mask_tensor = ops.convert_to_tensor(padding_mask)
        actual_output = ops.convert_to_numpy(
            layer(inputs, padding_mask=padding_mask_tensor)
        )

        self.assertAllClose(actual_output, expected_output)

    def test_config_serialization(self):
        """Tests that the layer can be successfully saved and loaded."""
        layer = Gemma4MeanPooling(name="mean_pooling_test")
        config = layer.get_config()
        new_layer = Gemma4MeanPooling.from_config(config)
        self.assertEqual(new_layer.name, layer.name)
        self.assertIsInstance(new_layer, Gemma4MeanPooling)


class Gemma4InterleaveEmbeddingsTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.max_images_per_prompt = 3
        self.num_vision_tokens_per_image = 2
        self.embedding_dim = 4
        self.seq_length = 10

        self.image_embeddings = np.random.randn(
            self.batch_size,
            self.max_images_per_prompt,
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
        reconstructed = ops.convert_to_numpy(
            Gemma4InterleaveEmbeddings(
                num_vision_tokens_per_image=self.num_vision_tokens_per_image,
            )(
                self.image_embeddings,
                self.text_embeddings,
                self.vision_indices,
            )
        )

        self.assertEqual(
            ops.shape(reconstructed),
            (self.batch_size, self.seq_length, self.embedding_dim),
        )

        # Index 0 should never be overwritten.
        self.assertAllEqual(
            reconstructed[:, 0, :], self.text_embeddings[:, 0, :]
        )

        # Verify vision tokens were interleaved in the first sample.
        self.assertAllEqual(
            reconstructed[0, 1:3],
            self.image_embeddings[0, 0],
        )

        # Verify vision tokens for the second sample.
        self.assertAllEqual(
            reconstructed[1, 3:5],
            self.image_embeddings[1, 0],
        )
        self.assertAllEqual(
            reconstructed[1, 7:9],
            self.image_embeddings[1, 1],
        )

    def test_non_vision_positions_unchanged(self):
        """Non-vision positions should retain their original text embeddings."""
        reconstructed = ops.convert_to_numpy(
            Gemma4InterleaveEmbeddings(
                num_vision_tokens_per_image=self.num_vision_tokens_per_image,
            )(
                self.image_embeddings,
                self.text_embeddings,
                self.vision_indices,
            )
        )
        # Positions 3-9 in sample 0 are not vision tokens; verify they match.
        self.assertAllEqual(
            reconstructed[0, 3:],
            self.text_embeddings[0, 3:],
        )

    def test_config_serialization(self):
        layer = Gemma4InterleaveEmbeddings(
            num_vision_tokens_per_image=4,
            name="interleave_test",
        )
        config = layer.get_config()
        new_layer = Gemma4InterleaveEmbeddings.from_config(config)
        self.assertEqual(
            new_layer.num_vision_tokens_per_image,
            layer.num_vision_tokens_per_image,
        )
