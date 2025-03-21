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

        self.mask = np.ones((self.batch_size, self.seq_length), dtype=bool)
        self.mask[0, : 3 * self.num_vision_tokens_per_image] = False
        self.mask[1, : self.num_vision_tokens_per_image] = False
        self.mask[1, 3 * self.num_vision_tokens_per_image :] = False

    def test_call(self):
        reconstructed = Gemma3InterleaveEmbeddings(
            image_max_length=self.image_max_length,
            num_vision_tokens_per_image=self.num_vision_tokens_per_image,
        )(
            self.image_embeddings,
            self.text_embeddings,
            self.mask,
        )

        self.assertEqual(
            ops.shape(reconstructed),
            (self.batch_size, self.seq_length, self.embedding_dim),
        )
