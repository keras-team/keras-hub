import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from keras_hub.src.models.qwen3_omni.qwen3_omni_video_converter import (
    Qwen3OmniVideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniVideoConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "min_pixels": 256 * 256,
            "max_pixels": 1024 * 1024,
        }

    def test_converter_basics(self):
        converter = Qwen3OmniVideoConverter(**self.init_kwargs)
        video = np.ones((4, 512, 512, 3), dtype=np.uint8) * 128
        output = converter(video)
        self.assertIn("patches", output)
        self.assertIn("grid_thw", output)
        patches = ops.convert_to_numpy(output["patches"])
        grid_thw = ops.convert_to_numpy(output["grid_thw"])
        # patches: (num_patches, temporal_patch_size, patch, patch, 3)
        self.assertEqual(patches.ndim, 5)
        self.assertEqual(patches.shape[1], 2)
        self.assertEqual(patches.shape[2], 16)
        self.assertEqual(patches.shape[3], 16)
        self.assertEqual(patches.shape[4], 3)
        # grid_thw: [T, H_grid, W_grid]
        self.assertEqual(grid_thw.shape, (3,))
        self.assertEqual(
            patches.shape[0],
            int(grid_thw[0]) * int(grid_thw[1]) * int(grid_thw[2]),
        )

    def test_temporal_padding_for_odd_frame_count(self):
        # 5 frames with temporal_patch_size=2 must be padded to 6, so grid_t=3.
        converter = Qwen3OmniVideoConverter(**self.init_kwargs)
        video = np.ones((5, 320, 320, 3), dtype=np.uint8) * 200
        output = converter(video)
        grid_thw = ops.convert_to_numpy(output["grid_thw"])
        self.assertEqual(int(grid_thw[0]), 3)

    def test_patch_stride_divisibility(self):
        converter = Qwen3OmniVideoConverter(**self.init_kwargs)
        video = np.random.randint(0, 255, (4, 400, 400, 3), dtype=np.uint8)
        output = converter(video)
        grid_thw = ops.convert_to_numpy(output["grid_thw"])
        merge = self.init_kwargs["spatial_merge_size"]
        self.assertEqual(int(grid_thw[1]) % merge, 0)
        self.assertEqual(int(grid_thw[2]) % merge, 0)

    def test_tf_and_eager_grid_parity(self):
        # Regression test: TF-graph path (used inside `tf.data.Dataset.map`)
        # must produce the same grid_thw and patch count as the eager path
        # across upscale, no-scale, and downscale regimes.
        converter = Qwen3OmniVideoConverter(**self.init_kwargs)

        test_cases = [
            (2, 64, 64),  # below min_pixels -> upscale
            (3, 320, 320),  # within range, odd temporal frame count
            (4, 400, 400),  # within range
            (2, 1500, 1500),  # above max_pixels -> downscale
        ]

        for frames, h, w in test_cases:
            video = np.ones((frames, h, w, 3), dtype=np.uint8) * 128

            eager_out = converter(video)
            eager_grid = ops.convert_to_numpy(eager_out["grid_thw"])
            eager_patches = ops.convert_to_numpy(eager_out["patches"])

            @tf.function
            def _tf_call(x):
                return converter(x)

            tf_out = _tf_call(tf.constant(video))
            tf_grid = tf_out["grid_thw"].numpy()
            tf_patches = tf_out["patches"].numpy()

            self.assertAllEqual(
                tf_grid,
                eager_grid,
                msg=f"grid_thw mismatch for {(frames, h, w)}",
            )
            self.assertEqual(
                tf_patches.shape,
                eager_patches.shape,
                msg=f"patch shape mismatch for {(frames, h, w)}",
            )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3OmniVideoConverter.presets:
            self.run_preset_test(
                cls=Qwen3OmniVideoConverter,
                preset=preset,
                input_data=np.ones((4, 224, 224, 3), dtype=np.uint8),
            )
