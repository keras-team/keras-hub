import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)


class Gemma4ImageConverterTest(tf.test.TestCase):
    def test_image_converter_range(self):
        converter = Gemma4ImageConverter(
            patch_size=4,
            max_soft_tokens=1,
            scale=[1 / 255.0, 1 / 255.0, 1 / 255.0],
            offset=[0.0, 0.0, 0.0],
        )

        # Minimum valid size: patch_size * pooling_kernel_size = 4 * 3 = 12
        img = np.random.randint(0, 256, size=(1, 12, 12, 3)).astype("float32")

        # Process image
        outputs = converter(img)

        pixel_values = ops.convert_to_numpy(outputs["pixel_values"])

        # Verify range is in [0, 1] (or close due to float precision)
        # Note: scaled by 1/255
        self.assertAllInRange(pixel_values, 0.0, 1.0)

    def test_aspect_ratio_resizing(self):
        converter = Gemma4ImageConverter(patch_size=4, max_soft_tokens=1)

        # Non-square input: 2:1 aspect ratio, dimensions are multiples of
        # patch_size (4) so the aspect-ratio resizer has room to adjust.
        img = np.random.randint(0, 256, size=(1, 24, 12, 3)).astype("float32")

        outputs = converter(img)

        # Resizing should preserve aspect ratio or adjust logic
        # Here we just verify it runs and outputs expected keys
        self.assertIn("pixel_values", outputs)
        self.assertIn("pixel_position_ids", outputs)


if __name__ == "__main__":
    tf.test.main()
