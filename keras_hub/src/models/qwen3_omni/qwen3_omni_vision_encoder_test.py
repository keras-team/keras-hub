import numpy as np

from keras_hub.src.models.qwen3_omni.qwen3_omni_vision_encoder import (
    Qwen3OmniVisionEncoder,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_vision_encoder import (
    Qwen3OmniVisionPatchEmbed,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniVisionEncoderTest(TestCase):
    def test_patch_embed_output_shape(self):
        patch_embed = Qwen3OmniVisionPatchEmbed(
            patch_size=2,
            temporal_patch_size=2,
            in_channels=3,
            embed_dim=32,
            dtype="float32",
        )
        pixel_values = np.random.rand(1, 2, 4, 4, 3).astype("float32")
        output = patch_embed(pixel_values)
        self.assertEqual(output.shape, (1, 4, 32))

    def test_encoder_output_shape(self):
        encoder = Qwen3OmniVisionEncoder(
            depth=2,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=49,
            deepstack_visual_indexes=[0],
            dtype="float32",
        )
        pixel_values = np.random.rand(1, 2, 4, 4, 3).astype("float32")
        grid_thw = np.array([[1, 2, 2]], dtype="int32")

        output = encoder({"pixel_values": pixel_values, "grid_thw": grid_thw})

        self.assertIsInstance(output, dict)
        self.assertIn("last_hidden_state", output)
        self.assertIn("pooler_output", output)
        self.assertIn("deepstack_features", output)
        self.assertEqual(output["pooler_output"].shape[-1], 16)

    def test_encoder_config_roundtrip(self):
        encoder = Qwen3OmniVisionEncoder(
            depth=2,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=16,
            num_position_embeddings=49,
            deepstack_visual_indexes=[0],
            dtype="float32",
        )
        config = encoder.get_config()
        restored = Qwen3OmniVisionEncoder.from_config(config)
        self.assertEqual(restored.depth, 2)
        self.assertEqual(restored.hidden_size, 32)
        self.assertEqual(restored.out_hidden_size, 16)
