import numpy as np
from keras import ops

from keras_hub.src.models.gemma3.gemma3_interleave_embeddings import (
    Gemma3InterleaveEmbeddings,
)
from keras_hub.src.tests.test_case import TestCase


class InterleaveEmbeddingsTest(TestCase):
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
