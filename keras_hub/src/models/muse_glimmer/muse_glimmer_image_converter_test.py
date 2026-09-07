import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    MuseGlimmerImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerImageConverterTest(TestCase):
    def test_convert(self):
        converter = MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            scale=1 / 255.0,
        )
        image = np.random.randint(0, 255, (16, 16, 3)).astype("float32")
        output = converter(image)
        grid_thw = output["grid_thw"]
        t, h, w = (int(v) for v in grid_thw)
        self.assertEqual(t, 1)
        num_patches = t * h * w
        patch_dim = 2 * 3 * 4 * 4  # patch_temporal * 3 * patch_size**2
        self.assertEqual(output["patches"].shape, (num_patches, patch_dim))

    def test_get_config(self):
        converter = MuseGlimmerImageConverter(patch_size=14)
        config = converter.get_config()
        restored = MuseGlimmerImageConverter.from_config(config)
        self.assertEqual(restored.patch_size, 14)
