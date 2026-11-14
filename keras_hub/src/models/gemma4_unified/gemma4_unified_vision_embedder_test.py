import numpy as np
from keras import ops

from keras_hub.src.models.gemma4_unified.gemma4_unified_vision_embedder import (
    Gemma4UnifiedVisionEmbedder,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4UnifiedVisionEmbedderTest(TestCase):
    EXPECTED = None

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

    def _seed_weights(self, embedder, rng):
        embedder.set_weights(
            [
                rng.standard_normal(w.shape).astype("float32") * 0.1
                for w in embedder.weights
            ]
        )

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

    def test_exact_output(self):
        embedder = self._make_embedder(
            hidden_dim=32,
            model_patch_size=48,
            mm_posemb_size=256,
            num_soft_tokens=16,
            pooling_kernel_size=3,
            patch_size=16,
        )
        input_dim = 48 * 48 * 3
        pv = np.ones((1, 1, 16, input_dim), dtype="float32")
        pos = np.zeros((1, 1, 16, 2), dtype="int32")
        for i in range(16):
            pos[0, 0, i, 0] = i % 4
            pos[0, 0, i, 1] = i // 4

        # Build first
        embedder({"pixel_values": pv, "pixel_position_ids": pos})

        # Set deterministic weights
        self._seed_weights(embedder, np.random.default_rng(0))

        output = embedder({"pixel_values": pv, "pixel_position_ids": pos})
        out_np = ops.convert_to_numpy(output)
        expected = [0.93682486, 0.4554378, -0.17883497, 0.6127433, -0.9761301]
        # Measured on CPU; 1e-3 covers TF32 drift in the GPU matrix (a x2
        # mutation moves these values by ~1).
        self.assertAllClose(out_np[0, 0, 0, :5], expected, atol=1e-3)

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
        out0 = ops.convert_to_numpy(output[0, 0, 0, :])
        out1 = ops.convert_to_numpy(output[0, 1, 0, :])
        np.testing.assert_allclose(out0, out1, rtol=1e-5)

    def test_num_vision_tokens_per_image(self):
        """Property should return num_soft_tokens."""
        embedder = self._make_embedder()
        self.assertEqual(embedder.num_vision_tokens_per_image, 16)

    def test_layer_norm_epsilon_matches_torch_default(self):
        # HF's `patch_ln1`/`patch_ln2`/`pos_norm` are `nn.LayerNorm(dim)` with
        # PyTorch's default eps 1e-5; only the RMSNorm reads `rms_norm_eps`.
        # On a near-flat patch row (std ~ 0.005, var ~ 2.5e-5) the choice of
        # eps changes 1/sqrt(var + eps) by ~16%, which is what sank the
        # 640x426 parity case (`patch_ln2` off by 6.47 on the 12B).
        embedder = self._make_embedder()
        input_dim = 48 * 48 * 3
        rng = np.random.default_rng(0)
        pv = (1.0 + 0.005 * rng.standard_normal((1, 1, 1, input_dim))).astype(
            "float32"
        )
        pos = np.zeros((1, 1, 1, 2), dtype="int32")
        embedder({"pixel_values": pv, "pixel_position_ids": pos})
        self._seed_weights(embedder, rng)
        output = ops.convert_to_numpy(
            embedder({"pixel_values": pv, "pixel_position_ids": pos})
        )

        def layer_norm(x, name):
            gamma, beta = embedder.get_layer(name).get_weights()
            mean = x.mean(-1, keepdims=True)
            var = x.var(-1, keepdims=True)
            return (x - mean) / np.sqrt(var + 1e-5) * gamma + beta

        x = pv.astype("float64")[0, 0]
        x = layer_norm(x, "patch_ln1")
        kernel, bias = embedder.get_layer("patch_dense").get_weights()
        x = layer_norm(x @ kernel + bias, "patch_ln2")
        x = x + embedder.get_layer("pos_embedding_x").get_weights()[0][0]
        x = x + embedder.get_layer("pos_embedding_y").get_weights()[0][0]
        x = layer_norm(x, "pos_norm")
        eps = embedder.layer_norm_epsilon
        x = x / np.sqrt((x**2).mean(-1, keepdims=True) + eps)
        x = x @ embedder.get_layer("embedding_projection").get_weights()[0]
        # 1e-3 covers float32/TF32 drift through three LayerNorms (as in
        # `test_exact_output`); the eps 1e-6 mutant moves values by O(0.1-1).
        self.assertAllClose(output[0, 0], x, atol=1e-3)

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
