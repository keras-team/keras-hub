import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.models.gemma4.gemma4_video_converter import (
    Gemma4VideoConverter,
)


class Gemma4VideoConverterTest(tf.test.TestCase):
    def test_video_converter_range(self):
        converter = Gemma4VideoConverter(
            patch_size=16,
            scale=[1 / 255.0, 1 / 255.0, 1 / 255.0],
            offset=[0.0, 0.0, 0.0],
            num_frames=8,
        )

        # Create a random video with values in [0, 255]
        # Shape: (batch, frames, height, width, channels)
        video = np.random.randint(0, 256, size=(1, 10, 224, 224, 3)).astype(
            "float32"
        )

        # Process video
        outputs = converter(video)

        pixel_values = ops.convert_to_numpy(outputs["pixel_values"])

        # Verify range is in [0, 1]
        self.assertAllInRange(pixel_values, 0.0, 1.0)

    def test_video_converter_shapes(self):
        converter = Gemma4VideoConverter(
            patch_size=16,
            num_frames=4,
            max_soft_tokens=70,
        )

        # Shape: (batch, frames, height, width, channels)
        video = np.random.randint(0, 256, size=(2, 10, 224, 224, 3)).astype(
            "float32"
        )

        outputs = converter(video)

        self.assertIn("pixel_values", outputs)
        self.assertIn("pixel_position_ids", outputs)

        # Check shapes
        # Max patches = 70 * 9 = 630
        # Patch pixels = 16 * 16 * 3 = 768
        self.assertEqual(ops.shape(outputs["pixel_values"]), (2, 4, 630, 768))
        self.assertEqual(
            ops.shape(outputs["pixel_position_ids"]), (2, 4, 630, 2)
        )

    def test_video_converter_list_input(self):
        converter = Gemma4VideoConverter(
            patch_size=16,
            num_frames=4,
            max_soft_tokens=70,
        )

        # List of videos
        video1 = np.random.randint(0, 256, size=(10, 224, 224, 3)).astype(
            "float32"
        )
        video2 = np.random.randint(0, 256, size=(8, 224, 224, 3)).astype(
            "float32"
        )
        videos = [video1, video2]

        outputs = converter(videos)

        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)

        # Check shapes of first video in list
        self.assertEqual(
            ops.shape(outputs[0]["pixel_values"]), (1, 4, 630, 768)
        )
        self.assertEqual(
            ops.shape(outputs[0]["pixel_position_ids"]), (1, 4, 630, 2)
        )


if __name__ == "__main__":
    tf.test.main()
