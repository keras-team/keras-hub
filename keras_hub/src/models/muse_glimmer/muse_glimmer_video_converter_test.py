import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    MuseGlimmerVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerVideoConverterTest(TestCase):
    def test_integer_inputs_round_after_resize(self):
        converter = MuseGlimmerVideoConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=1,
            fps=2.0,
            num_frames=2,
            max_video_frame_tokens=1,
            scale=2 / 255.0,
            offset=-1.0,
            interpolation="bilinear",
            antialias=True,
        )
        video = np.arange(2 * 4 * 4 * 3, dtype="uint8").reshape(2, 4, 4, 3)

        output = converter(video)
        normalized = (output["patches"] + 1.0) / (2.0 / 255.0)

        self.assertAllClose(normalized, np.round(normalized))

    def test_patch_layout_is_temporal_channel_first(self):
        converter = MuseGlimmerVideoConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=1,
            fps=2.0,
            num_frames=2,
            max_video_frame_tokens=16,
            interpolation="nearest",
        )
        video = np.arange(2 * 4 * 4 * 3, dtype="float32").reshape(2, 4, 4, 3)

        output = converter(video)

        expected_patch = np.array(
            [
                0,
                3,
                12,
                15,
                1,
                4,
                13,
                16,
                2,
                5,
                14,
                17,
                48,
                51,
                60,
                63,
                49,
                52,
                61,
                64,
                50,
                53,
                62,
                65,
            ],
            dtype="float32",
        )
        self.assertAllClose(output["patches"][0], expected_patch)

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

    def test_serialization(self):
        converter = MuseGlimmerVideoConverter(patch_size=14)
        self.run_serialization_test(converter)
