import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.models.gemma4_unified.gemma4_unified_image_converter import (
    Gemma4UnifiedImageConverter,
)


class Gemma4UnifiedImageConverterTest(tf.test.TestCase):
    """Tests for Gemma4UnifiedImageConverter."""

    def _make_converter(self, **kwargs):
        defaults = {
            "patch_size": 16,
            "max_soft_tokens": 280,
            "pooling_kernel_size": 3,
        }
        defaults.update(kwargs)
        return Gemma4UnifiedImageConverter(**defaults)

    def test_output_keys(self):
        """Converter should return pixel_values and pixel_position_ids."""
        converter = self._make_converter()
        img = np.random.rand(1, 768, 768, 3).astype("float32") * 255
        outputs = converter(img)
        self.assertIn("pixel_values", outputs)
        self.assertIn("pixel_position_ids", outputs)

    def test_output_shapes(self):
        """Output tensors should have the expected shapes."""
        converter = self._make_converter()
        img = np.random.rand(1, 768, 768, 3).astype("float32") * 255
        outputs = converter(img)

        pv = ops.convert_to_numpy(outputs["pixel_values"])
        pos = ops.convert_to_numpy(outputs["pixel_position_ids"])

        # (batch, max_soft_tokens, model_patch_dim)
        self.assertEqual(pv.shape[0], 1)
        self.assertEqual(pv.shape[1], 280)
        # model_patch_dim = (pooling_kernel_size * patch_size)^2 * 3
        expected_dim = (3 * 16) ** 2 * 3  # 6912
        self.assertEqual(pv.shape[2], expected_dim)

        # (batch, max_soft_tokens, 2)
        self.assertEqual(pos.shape, (1, 280, 2))

    def test_padding_positions_are_negative(self):
        """Padding positions should be -1, valid positions >= 0."""
        converter = self._make_converter()
        img = np.random.rand(1, 768, 768, 3).astype("float32") * 255
        outputs = converter(img)
        pos = ops.convert_to_numpy(outputs["pixel_position_ids"])

        # Valid positions should be >= 0
        valid_mask = ~np.all(pos == -1, axis=-1)
        if valid_mask.any():
            valid_pos = pos[valid_mask]
            self.assertTrue(np.all(valid_pos >= 0))

    def test_no_mixed_padding_corruption(self):
        """Valid positions should never be -1 due to mixed kernel groups."""
        converter = self._make_converter()
        img = np.random.rand(1, 768, 768, 3).astype("float32") * 255
        outputs = converter(img)
        pos = ops.convert_to_numpy(outputs["pixel_position_ids"])[0]

        padding_mask = np.all(pos == -1, axis=-1)
        valid_pos = pos[~padding_mask]
        # All valid positions must be >= 0 (no corruption from padding)
        self.assertTrue(
            np.all(valid_pos >= 0),
            f"Found negative coords in valid positions: "
            f"{valid_pos[valid_pos < 0]}",
        )

    def test_tf_data_path(self):
        """Converter should work inside tf.data.Dataset.map."""
        converter = self._make_converter()
        imgs = np.random.rand(2, 768, 768, 3).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(imgs).batch(2).map(converter)
        for batch in ds.take(1):
            pv = batch["pixel_values"]
            pos = batch["pixel_position_ids"]
            self.assertEqual(pv.shape[0], 2)
            self.assertEqual(pv.shape[1], 280)
            self.assertEqual(pos.shape, (2, 280, 2))

    def test_get_config_round_trip(self):
        """get_config / from_config should reproduce identical parameters."""
        converter = self._make_converter()
        config = converter.get_config()

        self.assertEqual(config["patch_size"], 16)
        self.assertEqual(config["max_soft_tokens"], 280)
        self.assertEqual(config["pooling_kernel_size"], 3)

        restored = Gemma4UnifiedImageConverter.from_config(config)
        self.assertEqual(restored.patch_size, 16)
        self.assertEqual(restored.max_soft_tokens, 280)
        self.assertEqual(restored.pooling_kernel_size, 3)

    def test_scale_offset(self):
        """Scale and offset should normalise pixel values."""
        converter = self._make_converter(
            scale=[1 / 255.0, 1 / 255.0, 1 / 255.0],
            offset=[0.0, 0.0, 0.0],
        )
        img = np.random.randint(0, 256, size=(1, 768, 768, 3)).astype("float32")
        outputs = converter(img)
        pv = ops.convert_to_numpy(outputs["pixel_values"])
        self.assertAllInRange(pv, -1.0, 1.1)


if __name__ == "__main__":
    tf.test.main()
