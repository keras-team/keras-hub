import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    MuseGlimmerVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerVideoConverterTest(TestCase):
    def test_convert(self):
        converter = MuseGlimmerVideoConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            fps=2.0,
            num_frames=8,
            max_video_frame_tokens=16,
            scale=1 / 255.0,
        )
        video = np.random.randint(0, 255, (4, 16, 16, 3)).astype("float32")
        output = converter(video)
        grid_t, grid_h, grid_w = (int(v) for v in output["grid_thw"])
        num_patches = grid_t * grid_h * grid_w
        patch_dim = 2 * 3 * 4 * 4  # patch_temporal * 3 * patch_size**2
        self.assertEqual(output["patches"].shape, (num_patches, patch_dim))

    def test_get_config(self):
        converter = MuseGlimmerVideoConverter(patch_size=14)
        config = converter.get_config()
        restored = MuseGlimmerVideoConverter.from_config(config)
        self.assertEqual(restored.patch_size, 14)
