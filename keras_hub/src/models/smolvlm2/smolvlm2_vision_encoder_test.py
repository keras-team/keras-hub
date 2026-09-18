import numpy as np
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_vision_encoder import (
    SmolVLM2VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


def _make_encoder(**overrides):
    """Create a small SmolVLM2VisionEncoder for testing."""
    kwargs = {
        "image_size": 32,
        "patch_size": 16,
        "hidden_dim": 64,
        "intermediate_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "num_channels": 3,
        "layer_norm_epsilon": 1e-6,
    }
    kwargs.update(overrides)
    return SmolVLM2VisionEncoder(**kwargs)


class SmolVLM2VisionEncoderTest(TestCase):
    def test_output_shape(self):
        """Single image produces (1, num_patches, hidden_dim) output."""
        encoder = _make_encoder()
        # image_size=32, patch_size=16 → 2x2 = 4 patches.
        pixel_values = np.random.rand(1, 32, 32, 3).astype("float32")
        output = encoder({"pixel_values": pixel_values})
        self.assertEqual(output.shape, (1, 4, 64))

    def test_batch_output_shape(self):
        """Batch of images produces correct shape."""
        encoder = _make_encoder()
        pixel_values = np.random.rand(3, 32, 32, 3).astype("float32")
        output = encoder({"pixel_values": pixel_values})
        self.assertEqual(output.shape, (3, 4, 64))

    def test_empty_batch(self):
        """A zero-length image batch is what text-only calls pass in."""
        encoder = _make_encoder()
        pixel_values = np.zeros((0, 32, 32, 3), dtype="float32")
        output = encoder({"pixel_values": pixel_values})
        self.assertEqual(output.shape, (0, 4, 64))

    def test_output_does_not_contain_nan(self):
        """Forward pass should produce finite outputs."""
        encoder = _make_encoder()
        pixel_values = np.random.rand(1, 32, 32, 3).astype("float32")
        output = ops.convert_to_numpy(encoder({"pixel_values": pixel_values}))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_num_patches_formula(self):
        """Verify num_patches = (image_size / patch_size)^2."""
        for image_size, patch_size, expected_patches in [
            (32, 16, 4),
            (32, 8, 16),
            (64, 16, 16),
        ]:
            encoder = _make_encoder(
                image_size=image_size, patch_size=patch_size
            )
            pixel_values = np.random.rand(1, image_size, image_size, 3).astype(
                "float32"
            )
            output = encoder({"pixel_values": pixel_values})
            self.assertEqual(output.shape[1], expected_patches)

    def test_wrong_image_size_raises(self):
        """Mismatched input resolution should be an actionable error."""
        encoder = _make_encoder()
        pixel_values = np.random.rand(1, 48, 48, 3).astype("float32")
        # The functional input spec rejects it first...
        with self.assertRaisesRegex(ValueError, "expected shape"):
            encoder({"pixel_values": pixel_values})
        # ...and the embedding layer explains it if called directly.
        with self.assertRaisesRegex(ValueError, "image_size"):
            encoder.vision_embeddings(pixel_values)

    def test_get_config_roundtrip(self):
        """get_config should return all constructor arguments."""
        encoder = _make_encoder(
            image_size=64,
            patch_size=8,
            hidden_dim=128,
            intermediate_dim=256,
            num_layers=3,
            num_heads=8,
        )
        self.run_serialization_test(encoder)
        cfg = encoder.get_config()
        self.assertEqual(cfg["image_size"], 64)
        self.assertEqual(cfg["patch_size"], 8)
        self.assertEqual(cfg["hidden_dim"], 128)
        self.assertEqual(cfg["intermediate_dim"], 256)
        self.assertEqual(cfg["num_layers"], 3)
        self.assertEqual(cfg["num_heads"], 8)
        self.assertEqual(cfg["num_channels"], 3)
        self.assertEqual(cfg["layer_norm_epsilon"], 1e-6)

    def test_different_hidden_dim(self):
        """Encoder with different hidden_dim produces correct output dim."""
        encoder = _make_encoder(hidden_dim=128, intermediate_dim=256)
        pixel_values = np.random.rand(1, 32, 32, 3).astype("float32")
        output = encoder({"pixel_values": pixel_values})
        self.assertEqual(output.shape, (1, 4, 128))

    def test_parameter_count_positive(self):
        """Encoder should have a non-trivial number of parameters."""
        encoder = _make_encoder()
        self.assertGreater(encoder.count_params(), 0)
