import numpy as np
import pytest
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

    def test_video_converter_list_input(self):
        converter = EmbeddingGemma2VideoConverter(
            patch_size=16,
            pooling_kernel_size=1,
            max_frames=2,
        )
        # Two clips with different frame counts.
        video1 = np.ones((3, 16, 16, 3), dtype="float32")
        video2 = np.ones((4, 16, 16, 3), dtype="float32")
        out = converter([video1, video2])
        self.assertIsInstance(out["pixel_values"], list)
        self.assertEqual(len(out["pixel_values"]), 2)
        for pv in out["pixel_values"]:
            self.assertEqual(ops.shape(pv)[0], 2)  # sampled frames

    def test_video_converter_indices_match_linspace(self):
        # The index formula avoids a Python branch on the frame count; it
        # must reproduce the reference (`arange` below `max_frames`,
        # truncated `linspace` above) for every frame count.
        for max_frames in (1, 2, 4, 8, 32):
            converter = EmbeddingGemma2VideoConverter(
                patch_size=16, pooling_kernel_size=1, max_frames=max_frames
            )
            for total in range(1, 257):
                if total <= max_frames:
                    expected = np.arange(total, dtype="int32")
                else:
                    expected = np.linspace(0, total - 1, max_frames).astype(
                        "int32"
                    )
                got = ops.convert_to_numpy(converter._get_indices(total))
                self.assertAllEqual(
                    got, expected, msg=f"{max_frames=} {total=}"
                )

    @pytest.mark.tf_only
    def test_video_converter_indices_symbolic(self):
        import tensorflow as tf

        converter = EmbeddingGemma2VideoConverter(
            patch_size=16, pooling_kernel_size=1, max_frames=4
        )

        @tf.function(input_signature=[tf.TensorSpec([None, 8, 8, 3])])
        def sample(video):
            return converter._get_indices(tf.shape(video)[0])

        self.assertAllEqual(sample(tf.ones((3, 8, 8, 3))), [0, 1, 2])
        self.assertAllEqual(sample(tf.ones((10, 8, 8, 3))), [0, 3, 6, 9])
