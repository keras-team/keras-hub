import numpy as np
from keras import ops

from keras_hub.src.models.gemma3.gemma3_interleave_embeddings import (
    Gemma3InterleaveEmbeddings,
)
from keras_hub.src.tests.test_case import TestCase


class InterleaveEmbeddingsTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_max_length = 3
        self.num_vision_tokens_per_image = 2
        self.embedding_dim = 4
        self.seq_length = 10

        self.image_embeddings = np.random.randn(
            self.batch_size * self.image_max_length,
            self.num_vision_tokens_per_image,
            self.embedding_dim,
        ).astype("float32")

        self.text_embeddings = np.random.randn(
            self.batch_size, self.seq_length, self.embedding_dim
        ).astype("float32")

        self.vision_indices = np.array(
            [
                list(range(0, 6)),
                list(range(self.seq_length, 2 + self.seq_length))
                + list(range(6 + self.seq_length, 2 * self.seq_length)),
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

        # Verify interleaving.
        self.assertAllEqual(
            reconstructed[0, :6],
            np.reshape(self.image_embeddings[:3], (6, self.embedding_dim)),
        )
        self.assertAllEqual(
            reconstructed[1, :2],
            np.reshape(self.image_embeddings[3:4], (2, self.embedding_dim)),
        )
        self.assertAllEqual(
            reconstructed[1, 6:],
            np.reshape(self.image_embeddings[4:], (4, self.embedding_dim)),
        )
