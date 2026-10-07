import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen3_omni.qwen3_omni_image_converter import (
    Qwen3OmniImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniImageConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "min_pixels": 256 * 256,
            "max_pixels": 1024 * 1024,
        }

    def test_converter_basics(self):
        converter = Qwen3OmniImageConverter(**self.init_kwargs)
        # Create dummy image
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        output = converter(image)
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
        # grid_thw: [T=1, H_grid, W_grid]
        self.assertEqual(grid_thw.shape, (3,))
        self.assertEqual(int(grid_thw[0]), 1)
        # num_patches must equal H_grid * W_grid.
        self.assertEqual(patches.shape[0], int(grid_thw[1]) * int(grid_thw[2]))

    def test_patch_stride_divisibility(self):
        converter = Qwen3OmniImageConverter(**self.init_kwargs)
        image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        output = converter(image)
        grid_thw = ops.convert_to_numpy(output["grid_thw"])
        # Grid dims must be a multiple of spatial_merge_size so the merger
        # can downsample cleanly.
        self.assertEqual(
            int(grid_thw[1]) % self.init_kwargs["spatial_merge_size"], 0
        )
        self.assertEqual(
            int(grid_thw[2]) % self.init_kwargs["spatial_merge_size"], 0
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3OmniImageConverter.presets:
            self.run_preset_test(
                cls=Qwen3OmniImageConverter,
                preset=preset,
                input_data=np.ones((224, 224, 3), dtype=np.uint8),
            )
