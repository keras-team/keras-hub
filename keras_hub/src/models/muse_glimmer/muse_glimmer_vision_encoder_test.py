import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_vision_encoder import (
    MuseGlimmerVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerVisionEncoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_layers": 2,
            "hidden_size": 8,
            "num_heads": 2,
            "intermediate_size": 16,
            "patch_size": 2,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "layer_types": ["window_attention", "full_attention"],
        }

    def test_vision_encoder_standalone(self):
        encoder = MuseGlimmerVisionEncoder(**self.init_kwargs)
        grid_thw = np.array([[1, 4, 4]], dtype="int32")
        total_patches = 1 * 4 * 4
        patch_dim = 2 * 3 * 2 * 2  # patch_temporal * 3 * patch_size**2
        pixel_values = np.random.randn(total_patches, patch_dim).astype(
            "float32"
        )
        output = encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor(grid_thw),
        )
        # 16 patches merged 2x2 -> 4 tokens, out_hidden_size = 8 * 2**2 = 32.
        self.assertEqual(ops.shape(output), (4, 32))

    def test_vision_encoder_multi_image(self):
        encoder = MuseGlimmerVisionEncoder(**self.init_kwargs)
        grid_thw = np.array([[1, 4, 4], [1, 2, 2]], dtype="int32")
        total_patches = 1 * 4 * 4 + 1 * 2 * 2
        patch_dim = 2 * 3 * 2 * 2
        pixel_values = np.random.randn(total_patches, patch_dim).astype(
            "float32"
        )
        output = encoder(
            ops.convert_to_tensor(pixel_values),
            ops.convert_to_tensor(grid_thw),
        )
        # (16 + 4) patches merged 2x2 -> 4 + 1 = 5 tokens.
        self.assertEqual(ops.shape(output), (5, 32))

    def test_get_config(self):
        encoder = MuseGlimmerVisionEncoder(**self.init_kwargs)
        config = encoder.get_config()
        restored = MuseGlimmerVisionEncoder.from_config(config)
        self.assertEqual(restored.hidden_size, encoder.hidden_size)
