import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.models.gemma4.gemma4_video_converter import (
    Gemma4VideoConverter,
)


class Gemma4VideoConverterTest(tf.test.TestCase):
    def test_video_converter_range(self):
        converter = Gemma4VideoConverter(
            patch_size=4,
            max_soft_tokens=1,
            scale=[1 / 255.0, 1 / 255.0, 1 / 255.0],
            offset=[0.0, 0.0, 0.0],
            num_frames=2,
        )

        # Shape: (batch, total_frames, height, width, channels)
        video = np.random.randint(0, 256, size=(1, 3, 12, 12, 3)).astype(
            "float32"
        )

        outputs = converter(video)
        pixel_values = ops.convert_to_numpy(outputs["pixel_values"])

        # Verify normalised values are in [0, 1]
        self.assertAllInRange(pixel_values, 0.0, 1.0)

    def test_video_converter_shapes(self):
        converter = Gemma4VideoConverter(
            patch_size=4,
            max_soft_tokens=1,
            num_frames=2,
        )

        # Shape: (batch, total_frames, height, width, channels)
        video = np.random.randint(0, 256, size=(2, 3, 12, 12, 3)).astype(
            "float32"
        )

        outputs = converter(video)

        self.assertIn("pixel_values", outputs)
        self.assertIn("pixel_position_ids", outputs)

        # max_patches = max_soft_tokens * pooling_kernel_size^2 = 1*9 = 9
        # patch_dim   = 3 * patch_size^2                        = 3*16 = 48
        self.assertEqual(
            ops.shape(outputs["pixel_values"]),
            (2, 2, 9, 48),
        )
        self.assertEqual(
            ops.shape(outputs["pixel_position_ids"]),
            (2, 2, 9, 2),
        )

    def test_video_converter_list_input(self):
        converter = Gemma4VideoConverter(
            patch_size=4,
            max_soft_tokens=1,
            num_frames=2,
        )

        # List of two videos with different frame counts.
        video1 = np.random.randint(0, 256, size=(3, 12, 12, 3)).astype(
            "float32"
        )
        video2 = np.random.randint(0, 256, size=(4, 12, 12, 3)).astype(
            "float32"
        )
        videos = [video1, video2]

        outputs = converter(videos)

        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)

        # Each video gets a batch dim of 1 added by _process_single_video.
        self.assertEqual(
            ops.shape(outputs[0]["pixel_values"]),
            (1, 2, 9, 48),
        )
        self.assertEqual(
            ops.shape(outputs[0]["pixel_position_ids"]),
            (1, 2, 9, 2),
        )


if __name__ == "__main__":
    tf.test.main()
