"""Tests for Llama3VisionEncoder with two-stage support."""

import numpy as np

from keras_hub.src.models.llama3.llama3_vision_encoder import (
    Llama3VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionEncoderTest(TestCase):
    def test_encoder_single_stage(self):
        """Test single-stage encoder (original architecture)."""
        self.run_layer_test(
            cls=Llama3VisionEncoder,
            init_kwargs={
                "hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 2,
                "intermediate_dim": 64,
                "patch_size": 14,
                "image_size": 28,
                "num_channels": 3,
            },
            input_data=np.random.uniform(size=(2, 28, 28, 3)).astype("float32"),
            expected_output_shape=(2, 4, 32),
            expected_num_trainable_weights=37,
            run_precision_checks=False,
        )

    def test_encoder_two_stage(self):
        """Test two-stage encoder (Meta's architecture)."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=4,  # Total for reference
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
            num_channels=3,
            local_layers=3,  # Two-stage mode
            global_layers=1,
        )

        images = np.random.uniform(size=(2, 28, 28, 3)).astype("float32")
        outputs = encoder(images)

        # Output shape: (batch, num_patches, hidden_dim)
        # num_patches = (28/14)^2 = 4
        self.assertEqual(outputs.shape, (2, 4, 32))
        self.assertTrue(encoder.is_two_stage)
        self.assertEqual(len(encoder.local_transformer_layers), 3)
        self.assertEqual(len(encoder.global_transformer_layers), 1)

    def test_valid_call(self):
        """Test forward pass logic."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
            num_channels=3,
        )

        images = np.random.uniform(size=(2, 28, 28, 3)).astype("float32")
        outputs = encoder(images)

        # Expected output: (Batch, Num_Patches, Hidden_Dim)
        self.assertEqual(outputs.shape, (2, 4, 32))

    def test_variable_batch_size(self):
        """Test dynamic batch size handling."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
        )

        encoder.build((None, 28, 28, 3))
        images = np.random.uniform(size=(5, 28, 28, 3)).astype("float32")
        outputs = encoder(images)
        self.assertEqual(outputs.shape, (5, 4, 32))

    def test_serialization(self):
        """Test config serialization."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=4,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
            local_layers=3,
            global_layers=1,
        )

        config = encoder.get_config()

        self.assertEqual(config["hidden_dim"], 32)
        self.assertEqual(config["local_layers"], 3)
        self.assertEqual(config["global_layers"], 1)

        # Recreate from config
        new_encoder = Llama3VisionEncoder(**config)
        self.assertTrue(new_encoder.is_two_stage)

    # ============================================================
    # New tests for missing coverage
    # ============================================================

    def test_freeze_local_encoder(self):
        """Test freezing local encoder in two-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=4,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
            local_layers=3,
            global_layers=1,
        )

        # Initially all should be trainable
        self.assertTrue(encoder.patch_embedding.trainable)
        self.assertTrue(encoder.position_embedding.trainable)

        # Freeze local encoder
        encoder.freeze_local_encoder()

        # Check that local components are frozen
        self.assertFalse(encoder.patch_embedding.trainable)
        self.assertFalse(encoder.position_embedding.trainable)
        for layer in encoder.local_transformer_layers:
            self.assertFalse(layer.trainable)

    def test_freeze_local_encoder_single_stage_raises_error(self):
        """Test that freeze_local_encoder raises error in single-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
        )

        with self.assertRaisesRegex(
            ValueError, "only available in two-stage mode"
        ):
            encoder.freeze_local_encoder()

    def test_freeze_global_encoder(self):
        """Test freezing global encoder in two-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=4,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
            local_layers=3,
            global_layers=1,
        )

        # Freeze global encoder
        encoder.freeze_global_encoder()

        # Check that global components are frozen
        self.assertFalse(encoder.layer_norm.trainable)
        for layer in encoder.global_transformer_layers:
            self.assertFalse(layer.trainable)

    def test_freeze_global_encoder_single_stage_raises_error(self):
        """Test that freeze_global_encoder raises error in single-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
        )

        with self.assertRaisesRegex(
            ValueError, "only available in two-stage mode"
        ):
            encoder.freeze_global_encoder()

    def test_freeze_all(self):
        """Test freezing entire encoder."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
        )

        # Initially trainable
        self.assertTrue(encoder.trainable)

        # Freeze all
        encoder.freeze_all()

        # Should be frozen
        self.assertFalse(encoder.trainable)

    def test_unfreeze_all_single_stage(self):
        """Test unfreezing all components in single-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
        )

        # Freeze first
        encoder.freeze_all()
        self.assertFalse(encoder.trainable)

        # Unfreeze
        encoder.unfreeze_all()

        # Should be trainable
        self.assertTrue(encoder.trainable)
        self.assertTrue(encoder.patch_embedding.trainable)
        self.assertTrue(encoder.position_embedding.trainable)
        self.assertTrue(encoder.layer_norm.trainable)
        for layer in encoder.transformer_layers:
            self.assertTrue(layer.trainable)

    def test_unfreeze_all_two_stage(self):
        """Test unfreezing all components in two-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=4,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
            local_layers=3,
            global_layers=1,
        )

        # Freeze first
        encoder.freeze_all()

        # Unfreeze
        encoder.unfreeze_all()

        # Check all are trainable
        self.assertTrue(encoder.trainable)
        for layer in encoder.local_transformer_layers:
            self.assertTrue(layer.trainable)
        for layer in encoder.global_transformer_layers:
            self.assertTrue(layer.trainable)

    def test_get_trainable_layers_summary_single_stage(self):
        """Test trainable layers summary in single-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
        )

        summary = encoder.get_trainable_layers_summary()

        self.assertIn("patch_embedding", summary)
        self.assertIn("position_embedding", summary)
        self.assertIn("layer_norm", summary)
        self.assertIn("transformer_layers", summary)
        self.assertTrue(summary["patch_embedding"])
        self.assertEqual(summary["transformer_layers"], "2/2")

    def test_get_trainable_layers_summary_two_stage(self):
        """Test trainable layers summary in two-stage mode."""
        encoder = Llama3VisionEncoder(
            hidden_dim=32,
            num_layers=4,
            num_heads=2,
            intermediate_dim=64,
            patch_size=14,
            image_size=28,
            local_layers=3,
            global_layers=1,
        )

        # Freeze local encoder
        encoder.freeze_local_encoder()

        summary = encoder.get_trainable_layers_summary()

        self.assertIn("local_layers", summary)
        self.assertIn("global_layers", summary)
        self.assertEqual(summary["local_layers"], "0/3")
        # Global layers should still be trainable
        self.assertEqual(summary["global_layers"], "1/1")
