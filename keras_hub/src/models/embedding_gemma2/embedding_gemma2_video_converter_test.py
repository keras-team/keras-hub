import numpy as np
from keras import ops

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_video_converter import (  # noqa: E501
    EmbeddingGemma2VideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2VideoConverterTest(TestCase):
    def test_video_converter_indices(self):
        converter = EmbeddingGemma2VideoConverter(
            patch_size=16,
            pooling_kernel_size=1,
            max_frames=32,
        )

        # Test keeping all frames (<= 32)
        total_frames = 10
        indices = converter._get_indices(total_frames)
        self.assertAllClose(indices, np.arange(10))

        # Test uniform sampling (> 32)
        total_frames = 64
        indices = converter._get_indices(total_frames)
        expected = np.linspace(0, 63, 32).astype("int32")
        self.assertAllClose(indices, expected)

    def test_video_converter_basics(self):
        converter = EmbeddingGemma2VideoConverter(
            patch_size=16,
            pooling_kernel_size=1,
            max_frames=4,
        )
        video = ops.ones((1, 10, 16, 16, 3))
        out = converter(video)
        self.assertIn("pixel_values", out)
        self.assertEqual(ops.shape(out["pixel_values"][0])[0], 4)  # frames
