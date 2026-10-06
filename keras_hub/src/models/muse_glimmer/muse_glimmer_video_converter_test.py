import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    MuseGlimmerVideoConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    _sample_frame_indices,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    _sample_frame_indices_tf,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf

# Expected indices of the HF `sample_frames` for `patch_temporal=2`. Each
# key is `(total_frames, fps, source_fps, num_frames)`. HF takes the indices
# from `torch.linspace(...).long()`.
HF_FRAME_INDICES = {
    # The sampled frame rate equals the source frame rate. `num_frames`
    # limits the count.
    (100, 1.0, 1.0, 8): [0, 14, 28, 42, 56, 70, 84, 99],
    (10, 1.0, 1.0, 4): [0, 3, 6, 9],
    # `num_frames=7` rounds down to 6 frames.
    (10, 1.0, 1.0, 7): [0, 1, 3, 5, 7, 9],
    # A float64 `np.linspace` gives 60 at position 50. HF gives 59.
    (115, 1.0, 1.0, 96): [
        *[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20],
        *[21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39],
        *[40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58],
        *[59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 78],
        *[79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97],
        *[98, 99, 100, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112],
        *[114],
    ],
    # The frame rate sets the count: `int(total * fps / source_fps)`.
    (240, 2.0, 24.0, 96): [
        *[0, 12, 25, 37, 50, 62, 75, 88, 100, 113, 125, 138, 150, 163],
        *[176, 188, 201, 213, 226, 239],
    ],
    (100, 2.0, 30.0, 96): [0, 19, 39, 59, 79, 99],
    # 5 frames round down to 4 frames.
    (59, 3.0, 30.0, 96): [0, 19, 38, 58],
    (48, 2.0, 24.0, 96): [0, 15, 31, 47],
    # HF drops the last frame of an odd frame count.
    (3, 1.0, 1.0, 8): [0, 2],
    # The count never goes below `patch_temporal`.
    (7, 2.0, 24.0, 96): [0, 6],
    (12, 2.0, 24.0, 96): [0, 11],
}


class MuseGlimmerVideoConverterTest(TestCase):
    def test_integer_inputs_round_after_resize(self):
        converter = MuseGlimmerVideoConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=1,
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

    def test_sample_frame_indices_match_hf(self):
        for key, expected in HF_FRAME_INDICES.items():
            total, fps, source_fps, num_frames = key
            self.assertEqual(
                _sample_frame_indices(total, num_frames, 2, fps, source_fps)
                .astype("int32")
                .tolist(),
                expected,
                msg=f"frames {key}",
            )

    def test_sample_frame_indices_keep_all_frames(self):
        # The converter keeps all frames if the sampled count equals the
        # frame count.
        self.assertIsNone(_sample_frame_indices(8, 8, 2, 1.0, 1.0))
        self.assertIsNone(_sample_frame_indices(1, 96, 2, 2.0, 24.0))

    def test_sample_frame_indices_tf_match_hf(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")

        def make_sample(num_frames, fps, source_fps):
            # `tf.function` fixes the arguments when it traces the graph.
            @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
            def sample(total):
                return _sample_frame_indices_tf(
                    total, num_frames, 2, fps, source_fps
                )

            return sample

        for key, expected in HF_FRAME_INDICES.items():
            total, fps, source_fps, num_frames = key
            sample = make_sample(num_frames, fps, source_fps)
            self.assertEqual(
                sample(total).numpy().tolist(), expected, msg=f"frames {key}"
            )
        # A video with one frame keeps that frame.
        self.assertEqual(make_sample(96, 2.0, 24.0)(1).numpy().tolist(), [0])

    def test_long_video_keeps_frames_spread_evenly(self):
        converter = MuseGlimmerVideoConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=1,
            fps=1.0,
            source_fps=1.0,
            num_frames=4,
            max_video_frame_tokens=16,
            interpolation="nearest",
        )
        video = np.ones((10, 4, 4, 3), dtype="float32")
        video *= np.arange(10, dtype="float32")[:, None, None, None]

        output = converter(video)

        # Frames 0, 3, 6 and 9 give two temporal patches. Each patch holds
        # 12 values per frame.
        self.assertEqual(int(output["grid_thw"][0]), 2)
        self.assertAllClose(output["patches"][0], [0.0] * 12 + [3.0] * 12)
        self.assertAllClose(output["patches"][-1], [6.0] * 12 + [9.0] * 12)

    def _frame_index_video(self):
        # Frame `i` holds the value `i`.
        video = np.ones((8, 4, 4, 3), dtype="float32")
        return video * np.arange(8, dtype="float32")[:, None, None, None]

    def _frame_index_converter(self):
        return MuseGlimmerVideoConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=1,
            fps=1.0,
            source_fps=1.0,
            num_frames=96,
            max_video_frame_tokens=16,
            interpolation="nearest",
        )

    def test_call_with_source_fps(self):
        converter = self._frame_index_converter()
        video = self._frame_index_video()

        # At `source_fps=1.0`, the converter keeps all 8 frames.
        output = converter(video)
        self.assertEqual(int(output["grid_thw"][0]), 4)

        # At `source_fps=2.0`, `int(8 * 1 / 2) = 4` frames: 0, 2, 4, 7.
        output = converter(video, source_fps=2.0)
        self.assertEqual(int(output["grid_thw"][0]), 2)
        self.assertAllClose(output["patches"][0], [0.0] * 12 + [2.0] * 12)
        self.assertAllClose(output["patches"][-1], [4.0] * 12 + [7.0] * 12)

    def test_call_with_source_fps_in_tf_function(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        converter = self._frame_index_converter()
        video = self._frame_index_video()

        @tf.function
        def convert(video, source_fps):
            return converter(video, source_fps=source_fps)

        output = convert(video, tf.constant(2.0, "float64"))
        self.assertEqual(int(output["grid_thw"][0]), 2)
        self.assertAllClose(output["patches"][0], [0.0] * 12 + [2.0] * 12)
        self.assertAllClose(output["patches"][-1], [4.0] * 12 + [7.0] * 12)

    def test_serialization(self):
        converter = MuseGlimmerVideoConverter(patch_size=14)
        self.run_serialization_test(converter)
