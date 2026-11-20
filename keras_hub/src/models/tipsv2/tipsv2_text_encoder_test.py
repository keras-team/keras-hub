"""Tests for TIPSv2 Text Encoder."""

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.tipsv2.tipsv2_text_encoder import TIPSv2TextEncoder
from keras_hub.src.tests.test_case import TestCase


class TIPSv2TextEncoderTest(TestCase):
    def setUp(self):
        self.hidden_dim = 32
        self.init_kwargs = {
            "vocabulary_size": 100,
            "embedding_dim": self.hidden_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": 2,
            "num_heads": 4,
            "intermediate_dim": 64,
            "max_sequence_length": 16,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 6), dtype="int32"),
            "padding_mask": ops.ones((2, 6), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=TIPSv2TextEncoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, self.hidden_dim),
        )

    def test_output_values(self):
        encoder = TIPSv2TextEncoder(**self.init_kwargs)
        outputs = ops.convert_to_numpy(encoder(self.input_data))
        # Output shape: (batch, hidden_dim).
        self.assertEqual(outputs.shape, (2, self.hidden_dim))
        # All values should be finite.
        self.assertTrue(np.all(np.isfinite(outputs)))

    def test_masking_effect(self):
        """Shorter sequences should produce different embeddings."""
        encoder = TIPSv2TextEncoder(**self.init_kwargs)

        # Same tokens, different padding.
        input1 = {
            "token_ids": np.array([[1, 2, 3, 4, 0, 0]], dtype="int32"),
            "padding_mask": np.array([[1, 1, 1, 1, 0, 0]], dtype="int32"),
        }
        input2 = {
            "token_ids": np.array([[1, 2, 0, 0, 0, 0]], dtype="int32"),
            "padding_mask": np.array([[1, 1, 0, 0, 0, 0]], dtype="int32"),
        }
        out1 = ops.convert_to_numpy(encoder(input1))
        out2 = ops.convert_to_numpy(encoder(input2))
        # Different padding lengths should produce different embeddings.
        self.assertFalse(np.allclose(out1, out2, atol=1e-5))

    def test_single_token(self):
        """Single valid token should still produce finite output."""
        encoder = TIPSv2TextEncoder(**self.init_kwargs)
        inputs = {
            "token_ids": np.array([[1, 0, 0, 0, 0, 0]], dtype="int32"),
            "padding_mask": np.array([[1, 0, 0, 0, 0, 0]], dtype="int32"),
        }
        out = ops.convert_to_numpy(encoder(inputs))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_no_scaling(self):
        """Test with scale_sqrt_depth=False."""
        init_kwargs = {**self.init_kwargs, "scale_sqrt_depth": False}
        encoder = TIPSv2TextEncoder(**init_kwargs)
        out = ops.convert_to_numpy(encoder(self.input_data))
        self.assertEqual(out.shape, (2, self.hidden_dim))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_get_config_roundtrip(self):
        encoder = TIPSv2TextEncoder(**self.init_kwargs)
        config = encoder.get_config()

        self.assertEqual(config["vocabulary_size"], 100)
        self.assertEqual(config["embedding_dim"], self.hidden_dim)
        self.assertEqual(config["hidden_dim"], self.hidden_dim)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["num_heads"], 4)
        self.assertEqual(config["intermediate_dim"], 64)
        self.assertEqual(config["max_sequence_length"], 16)
        self.assertTrue(config["scale_sqrt_depth"])

        # Roundtrip.
        restored = TIPSv2TextEncoder.from_config(config)
        out = ops.convert_to_numpy(restored(self.input_data))
        self.assertEqual(out.shape, (2, self.hidden_dim))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=TIPSv2TextEncoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.skipTest("Presets are not uploaded yet.")
        self.run_preset_test(
            cls=TIPSv2TextEncoder,
            preset="tipsv2_b14",
            input_data={
                "token_ids": ops.ones((1, 64), dtype="int32"),
                "padding_mask": ops.ones((1, 64), dtype="int32"),
            },
            expected_output_shape=(1, 768),
        )
