import numpy as np

from keras_hub.src.models.gemma4.gemma4_unified_vision_embedder import (
    Gemma4UnifiedVisionEmbedder,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4UnifiedVisionEmbedderTest(TestCase):
    """Tests for Gemma4UnifiedVisionEmbedder."""

    def _make_embedder(self, hidden_dim=32, mm_posemb_size=256, **kwargs):
        defaults = {
            "hidden_dim": hidden_dim,
            "model_patch_size": 48,
            "mm_posemb_size": mm_posemb_size,
            "num_soft_tokens": 16,
            "pooling_kernel_size": 3,
            "patch_size": 16,
            "dtype": "float32",
        }
        defaults.update(kwargs)
        return Gemma4UnifiedVisionEmbedder(**defaults)

    def test_output_shape(self):
        """Embedder produces (batch, n_images, num_soft_tokens, hidden_dim)."""
        embedder = self._make_embedder()
        input_dim = 48 * 48 * 3  # model_patch_size^2 * 3
        pv = np.random.rand(1, 1, 16, input_dim).astype("float32")
        pos = np.zeros((1, 1, 16, 2), dtype="int32")
        for i in range(16):
            pos[0, 0, i, 0] = i % 4
            pos[0, 0, i, 1] = i // 4

        output = embedder({"pixel_values": pv, "pixel_position_ids": pos})
        self.assertEqual(output.shape, (1, 1, 16, 32))

    def test_all_padding_input(self):
        """All-padding input (pos=-1) should not crash."""
        embedder = self._make_embedder()
        input_dim = 48 * 48 * 3
        pv = np.random.rand(1, 1, 16, input_dim).astype("float32")
        pos = np.full((1, 1, 16, 2), -1, dtype="int32")

        output = embedder({"pixel_values": pv, "pixel_position_ids": pos})
        self.assertEqual(output.shape, (1, 1, 16, 32))

    def test_mixed_aspect_batch(self):
        """Batching images with different aspect ratios should work."""
        embedder = self._make_embedder()
        input_dim = 48 * 48 * 3
        pv = np.random.rand(1, 2, 16, input_dim).astype("float32")
        pos = np.full((1, 2, 16, 2), -1, dtype="int32")

        # Image 1: wide (8 cols x 2 rows = 16 patches)
        for i in range(16):
            pos[0, 0, i, 0] = i % 8
            pos[0, 0, i, 1] = i // 8
        # Image 2: tall (2 cols x 8 rows = 16 patches)
        for i in range(16):
            pos[0, 1, i, 0] = i % 2
            pos[0, 1, i, 1] = i // 2

        output = embedder({"pixel_values": pv, "pixel_position_ids": pos})
        self.assertEqual(output.shape, (1, 2, 16, 32))

    def test_factorized_pos_embedding(self):
        """Same (x,y) should get the same pos embedding in different images."""
        embedder = self._make_embedder()
        input_dim = 48 * 48 * 3
        # Use identical pixel values so only pos embedding differs
        pv_data = np.zeros((1, 2, 16, input_dim), dtype="float32")
        pos = np.full((1, 2, 16, 2), -1, dtype="int32")

        # Both images: position (3, 5) at index 0
        pos[0, 0, 0, :] = [3, 5]
        pos[0, 1, 0, :] = [3, 5]

        output = embedder({"pixel_values": pv_data, "pixel_position_ids": pos})
        # Same input + same position → same output
        out0 = np.array(output[0, 0, 0, :])
        out1 = np.array(output[0, 1, 0, :])
        np.testing.assert_allclose(out0, out1, rtol=1e-5)

    def test_num_vision_tokens_per_image(self):
        """Property should return num_soft_tokens."""
        embedder = self._make_embedder()
        self.assertEqual(embedder.num_vision_tokens_per_image, 16)

    def test_get_config_round_trip(self):
        """get_config / from_config should produce identical model."""
        embedder = self._make_embedder()
        config = embedder.get_config()

        self.assertEqual(config["hidden_dim"], 32)
        self.assertEqual(config["model_patch_size"], 48)
        self.assertEqual(config["num_soft_tokens"], 16)

        restored = Gemma4UnifiedVisionEmbedder.from_config(config)
        self.assertEqual(restored.hidden_dim, 32)
        self.assertEqual(restored.num_soft_tokens, 16)

    def test_saved_model(self):
        """Model saving and loading round-trip."""
        input_dim = 48 * 48 * 3
        input_data = {
            "pixel_values": np.ones([1, 1, 16, input_dim], dtype="float32"),
            "pixel_position_ids": np.zeros([1, 1, 16, 2], dtype="int32"),
        }
        self.run_model_saving_test(
            cls=Gemma4UnifiedVisionEmbedder,
            init_kwargs={
                "hidden_dim": 32,
                "model_patch_size": 48,
                "mm_posemb_size": 256,
                "num_soft_tokens": 16,
                "pooling_kernel_size": 3,
                "patch_size": 16,
                "dtype": "float32",
            },
            input_data=input_data,
        )
