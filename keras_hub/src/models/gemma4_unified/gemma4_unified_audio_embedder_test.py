import numpy as np

from keras_hub.src.models.gemma4_unified.gemma4_unified_audio_embedder import (
    Gemma4UnifiedAudioEmbedder,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma4UnifiedAudioEmbedderTest(TestCase):
    """Tests for Gemma4UnifiedAudioEmbedder."""

    def _make_embedder(self, hidden_dim=32, audio_embed_dim=16, **kwargs):
        defaults = {
            "hidden_dim": hidden_dim,
            "audio_embed_dim": audio_embed_dim,
            "dtype": "float32",
        }
        defaults.update(kwargs)
        return Gemma4UnifiedAudioEmbedder(**defaults)

    def test_output_shape(self):
        """Embedder should project audio_embed_dim → hidden_dim."""
        embedder = self._make_embedder()
        audio_features = np.random.rand(2, 10, 16).astype("float32")
        audio_mask = np.ones((2, 10), dtype="bool")
        output = embedder(audio_features, audio_mask)
        self.assertEqual(output.shape, (2, 10, 32))

    def test_single_sample(self):
        """Single-sample batched input should work."""
        embedder = self._make_embedder()
        audio_features = np.random.rand(1, 10, 16).astype("float32")
        audio_mask = np.ones((1, 10), dtype="bool")
        output = embedder(audio_features, audio_mask)
        self.assertEqual(output.shape, (1, 10, 32))

    def test_output_is_finite(self):
        """Output should not contain NaN or Inf."""
        import keras

        embedder = self._make_embedder()
        audio_features = np.random.rand(2, 10, 16).astype("float32")
        audio_mask = np.ones((2, 10), dtype="bool")
        output = embedder(audio_features, audio_mask)
        self.assertFalse(
            bool(keras.ops.any(keras.ops.isnan(output))),
            "Output contains NaN",
        )
        self.assertFalse(
            bool(keras.ops.any(keras.ops.isinf(output))),
            "Output contains Inf",
        )

    def test_get_config_round_trip(self):
        """get_config / from_config should reproduce identical parameters."""
        embedder = self._make_embedder()
        config = embedder.get_config()

        self.assertEqual(config["hidden_dim"], 32)
        self.assertEqual(config["audio_embed_dim"], 16)

        restored = Gemma4UnifiedAudioEmbedder.from_config(config)
        self.assertEqual(restored.hidden_dim, 32)
        self.assertEqual(restored.audio_embed_dim, 16)

    def test_build_is_called(self):
        """Embedder should be built after first call."""
        embedder = self._make_embedder()
        audio_features = np.random.rand(2, 10, 16).astype("float32")
        audio_mask = np.ones((2, 10), dtype="bool")
        embedder(audio_features, audio_mask)
        self.assertTrue(embedder.built)

    def test_custom_layer_norm_epsilon(self):
        """Custom layer_norm_epsilon should be stored correctly."""
        embedder = self._make_embedder(layer_norm_epsilon=1e-5)
        self.assertAlmostEqual(embedder.layer_norm_epsilon, 1e-5)

    def test_with_partial_mask(self):
        """Partial audio mask should not cause errors.

        Masked positions should be exactly zero.
        """
        embedder = self._make_embedder()
        audio_features = np.ones((2, 10, 16), dtype="float32")
        audio_mask = np.ones((2, 10), dtype="bool")

        # Build first
        embedder(audio_features, audio_mask)

        # Set deterministic weights
        rng = np.random.default_rng(0)
        embedder.set_weights(
            [
                rng.standard_normal(w.shape).astype("float32") * 0.1
                for w in embedder.weights
            ]
        )

        output = embedder(audio_features, audio_mask)
        self.assertEqual(output.shape, (2, 10, 32))

        import keras

        out_np = keras.ops.convert_to_numpy(output)
        expected = [
            -0.049372494,
            0.10350018,
            -0.15298781,
            0.28178325,
            -0.14632004,
        ]
        # Measured on CPU; 1e-3 covers TF32 drift in the GPU matrix (a x2
        # mutation moves these values by ~1).
        self.assertAllClose(out_np[0, 0, :5], expected, atol=1e-3)
