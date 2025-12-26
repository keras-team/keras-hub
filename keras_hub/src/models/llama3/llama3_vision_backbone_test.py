"""Tests for Llama3VisionBackbone with Cross-Attention."""

import numpy as np
import pytest

from keras_hub.src.models.llama3.llama3_backbone import Llama3BackboneConfig
from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)
from keras_hub.src.models.llama3.llama3_vision_config import Llama3VisionConfig
from keras_hub.src.models.llama3.llama3_vision_config import (
    Llama3VisionEncoderConfig,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionBackboneTest(TestCase):
    def setUp(self):
        # 1. Vision Config
        self.vision_config = Llama3VisionEncoderConfig(
            hidden_dim=16,
            num_layers=2,  # Need at least 2 layers for cross-attention test
            num_heads=2,
            intermediate_dim=32,
            patch_size=4,
            image_size=16,  # 16x16 / 4x4 = 16 patches
        )

        # 2. Text Config (hidden_dim matches what projector outputs)
        self.text_config = Llama3BackboneConfig(
            vocabulary_size=100,
            num_layers=4,  # Need multiple layers for cross-attention positions
            num_query_heads=2,
            num_key_value_heads=2,
            hidden_dim=16,
            intermediate_dim=32,
        )

        # 3. Main Config with cross-attention at layer 1
        self.config = Llama3VisionConfig(
            vision_encoder_config=self.vision_config,
            text_config=self.text_config,
            cross_attention_layers=[1, 3],  # Cross-attention at layers 1 and 3
        )

    def test_backbone_call(self):
        """Test forward pass produces correct output shape."""
        backbone = Llama3VisionBackbone(config=self.config)

        # Input Data
        batch_size = 2
        seq_len = 10
        images = np.random.uniform(size=(batch_size, 16, 16, 3)).astype(
            "float32"
        )
        token_ids = np.ones((batch_size, seq_len), dtype="int32")
        padding_mask = np.ones((batch_size, seq_len), dtype="bool")

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        outputs = backbone(inputs)

        # Output shape: (batch, seq_len, hidden_dim)
        # With cross-attention, output matches TEXT sequence length
        # (not concatenated)
        self.assertEqual(outputs.shape, (batch_size, seq_len, 16))

    def test_cross_attention_blocks_created(self):
        """Verify cross-attention blocks are created at specified layers."""
        backbone = Llama3VisionBackbone(config=self.config)

        # Should have cross-attention blocks at layers 1 and 3
        self.assertIn(1, backbone.cross_attention_blocks)
        self.assertIn(3, backbone.cross_attention_blocks)
        self.assertNotIn(0, backbone.cross_attention_blocks)
        self.assertNotIn(2, backbone.cross_attention_blocks)

    def test_no_cross_attention(self):
        """Test backbone with empty cross-attention layers."""
        config = Llama3VisionConfig(
            vision_encoder_config=self.vision_config,
            text_config=self.text_config,
            cross_attention_layers=[],  # No cross-attention
        )
        backbone = Llama3VisionBackbone(config=config)

        batch_size = 2
        seq_len = 10
        images = np.random.uniform(size=(batch_size, 16, 16, 3)).astype(
            "float32"
        )
        token_ids = np.ones((batch_size, seq_len), dtype="int32")
        padding_mask = np.ones((batch_size, seq_len), dtype="bool")

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        outputs = backbone(inputs)
        self.assertEqual(outputs.shape, (batch_size, seq_len, 16))

    def test_two_stage_encoder(self):
        """Test backbone with two-stage vision encoder."""
        vision_config = Llama3VisionEncoderConfig(
            hidden_dim=16,
            num_layers=4,
            num_heads=2,
            intermediate_dim=32,
            patch_size=4,
            image_size=16,
            local_layers=3,  # Two-stage: 3 local + 1 global
            global_layers=1,
        )

        config = Llama3VisionConfig(
            vision_encoder_config=vision_config,
            text_config=self.text_config,
            cross_attention_layers=[1],
        )

        backbone = Llama3VisionBackbone(config=config)

        batch_size = 2
        seq_len = 8
        images = np.random.uniform(size=(batch_size, 16, 16, 3)).astype(
            "float32"
        )
        token_ids = np.ones((batch_size, seq_len), dtype="int32")
        padding_mask = np.ones((batch_size, seq_len), dtype="bool")

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        outputs = backbone(inputs)
        self.assertEqual(outputs.shape, (batch_size, seq_len, 16))

    @pytest.mark.large
    def test_backbone_serialization(self):
        """Test model serialization via save/load."""
        import os

        import keras

        backbone = Llama3VisionBackbone(config=self.config)

        images = np.random.uniform(size=(2, 16, 16, 3)).astype("float32")
        token_ids = np.ones((2, 10), dtype="int32")
        padding_mask = np.ones((2, 10), dtype="bool")

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        original_output = backbone(inputs)

        # Save to disk
        path = os.path.join(self.get_temp_dir(), "model.keras")
        backbone.save(path)

        # Load from disk
        restored_backbone = keras.models.load_model(path)

        # Verify restored model produces same output
        restored_output = restored_backbone(inputs)
        self.assertAllClose(
            original_output, restored_output, atol=1e-5, rtol=1e-5
        )

        # Verify config was restored
        self.assertEqual(
            restored_backbone.config.vision_encoder_config.hidden_dim,
            self.config.vision_encoder_config.hidden_dim,
        )
        self.assertEqual(
            restored_backbone.cross_attention_layers,
            self.config.cross_attention_layers,
        )

    # ============================================================
    # New tests for missing coverage
    # ============================================================

    def test_missing_vision_encoder_config_raises_error(self):
        """Test that missing vision_encoder_config raises ValueError."""
        config = Llama3VisionConfig(
            vision_encoder_config=None,
            text_config=self.text_config,
        )
        config.vision_encoder_config = None  # Force None

        with self.assertRaisesRegex(
            ValueError, "`vision_encoder_config` must be provided"
        ):
            Llama3VisionBackbone(config=config)

    def test_missing_text_config_raises_error(self):
        """Test that missing text_config raises ValueError."""
        # Create a minimal config with None text_config
        class BadConfig:
            pass
        
        bad_config = BadConfig()
        bad_config.vision_encoder_config = self.vision_config
        bad_config.text_config = None
        bad_config.cross_attention_layers = [1]
        bad_config.dtype = None
        
        with self.assertRaisesRegex(
            ValueError, "`text_config` must be provided"
        ):
            Llama3VisionBackbone(config=bad_config)

    def test_dict_text_config(self):
        """Test initialization with dict-based text_config."""
        text_config_dict = {
            "vocabulary_size": 100,
            "num_layers": 4,
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
        }
        
        config = Llama3VisionConfig(
            vision_encoder_config=self.vision_config,
            text_config=text_config_dict,
            cross_attention_layers=[1],
        )
        
        backbone = Llama3VisionBackbone(config=config)
        
        # Verify it works
        batch_size = 2
        seq_len = 10
        images = np.random.uniform(size=(batch_size, 16, 16, 3)).astype(
            "float32"
        )
        token_ids = np.ones((batch_size, seq_len), dtype="int32")
        padding_mask = np.ones((batch_size, seq_len), dtype="bool")

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        outputs = backbone(inputs)
        self.assertEqual(outputs.shape, (batch_size, seq_len, 16))

    def test_from_config(self):
        """Test from_config class method."""
        backbone = Llama3VisionBackbone(config=self.config)
        config_dict = backbone.get_config()
        
        # Recreate from config
        restored_backbone = Llama3VisionBackbone.from_config(config_dict)
        
        # Verify config matches
        self.assertEqual(
            restored_backbone.config.vision_encoder_config.hidden_dim,
            self.config.vision_encoder_config.hidden_dim,
        )

    def test_freeze_vision_encoder(self):
        """Test freezing the vision encoder."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Initially trainable
        self.assertTrue(backbone.vision_encoder.trainable)
        
        # Freeze
        backbone.freeze_vision_encoder()
        
        # Should now be frozen
        self.assertFalse(backbone.vision_encoder.trainable)

    def test_unfreeze_all(self):
        """Test unfreezing all components."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Freeze everything first
        backbone.freeze_vision_encoder()
        backbone.freeze_text_backbone()
        backbone.freeze_cross_attention()
        
        # Unfreeze all
        backbone.unfreeze_all()
        
        # Everything should be trainable again
        self.assertTrue(backbone.vision_encoder.trainable)
        self.assertTrue(backbone.text_backbone.trainable)
        for layer_idx, ca_block in backbone.cross_attention_blocks.items():
            self.assertTrue(ca_block.trainable)

    def test_unfreeze_all_without_encoder_method(self):
        """Test unfreeze_all when vision_encoder lacks unfreeze_all method."""
        import keras
        
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Create a simple layer without unfreeze_all method
        simple_layer = keras.layers.Dense(16, name="simple_vision_encoder")
        
        # Replace vision_encoder temporarily
        original_encoder = backbone.vision_encoder
        backbone.vision_encoder = simple_layer
        
        try:
            # This should work without calling vision_encoder.unfreeze_all()
            # because simple_layer doesn't have that method
            backbone.unfreeze_all()
            
            # Basic unfreezing should still work
            self.assertTrue(backbone.vision_encoder.trainable)
        finally:
            # Restore original encoder
            backbone.vision_encoder = original_encoder

    def test_freeze_text_backbone(self):
        """Test freezing the text backbone."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Initially trainable
        self.assertTrue(backbone.text_backbone.trainable)
        
        # Freeze
        backbone.freeze_text_backbone()
        
        # Should now be frozen
        self.assertFalse(backbone.text_backbone.trainable)


    def test_freeze_cross_attention(self):
        """Test freezing cross-attention blocks."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Initially trainable
        for layer_idx, ca_block in backbone.cross_attention_blocks.items():
            self.assertTrue(ca_block.trainable)
        
        # Freeze
        backbone.freeze_cross_attention()
        
        # Should now be frozen
        for layer_idx, ca_block in backbone.cross_attention_blocks.items():
            self.assertFalse(ca_block.trainable)

    def test_freeze_for_vision_adapter_training(self):
        """Test freeze_for_vision_adapter_training method."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Apply adapter training freeze pattern
        backbone.freeze_for_vision_adapter_training()
        
        # Vision encoder should be frozen
        self.assertFalse(backbone.vision_encoder.trainable)
        # Text backbone should be frozen
        self.assertFalse(backbone.text_backbone.trainable)
        # Vision projector should be trainable
        self.assertTrue(backbone.vision_projector.trainable)
        # Cross-attention should be trainable
        for layer_idx, ca_block in backbone.cross_attention_blocks.items():
            self.assertTrue(ca_block.trainable)

    def test_freeze_for_lora_training(self):
        """Test freeze_for_lora_training method."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Apply LoRA freeze pattern
        backbone.freeze_for_lora_training()
        
        # Everything should be frozen
        self.assertFalse(backbone.vision_encoder.trainable)
        self.assertFalse(backbone.vision_projector.trainable)
        self.assertFalse(backbone.text_backbone.trainable)
        for layer_idx, ca_block in backbone.cross_attention_blocks.items():
            self.assertFalse(ca_block.trainable)

    def test_invalid_text_config_type_raises_error(self):
        """Test that invalid text_config type raises ValueError."""
        # Create a config with text_config that is neither dict
        # nor has get_config
        class BadConfig:
            pass
        
        bad_config = BadConfig()
        bad_config.vision_encoder_config = self.vision_config
        # Not dict, not object with get_config
        bad_config.text_config = "invalid_string"
        bad_config.cross_attention_layers = [1]
        bad_config.dtype = None
        
        with self.assertRaisesRegex(
            ValueError, "text_config must be either a dict or have a"
        ):
            Llama3VisionBackbone(config=bad_config)

    def test_get_trainable_summary(self):
        """Test get_trainable_summary method."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        summary = backbone.get_trainable_summary()
        
        # Check structure
        self.assertIn("vision_encoder", summary)
        self.assertIn("vision_projector", summary)
        self.assertIn("text_backbone", summary)
        self.assertIn("cross_attention", summary)
        self.assertIn("total", summary)
        
        # Check vision_encoder info
        self.assertTrue(summary["vision_encoder"]["trainable"])
        self.assertGreater(summary["vision_encoder"]["params"], 0)
        
        # Check total info
        self.assertIn("trainable_params", summary["total"])
        self.assertIn("total_params", summary["total"])
        self.assertIn("trainable_ratio", summary["total"])

    def test_get_trainable_summary_frozen_components(self):
        """Test get_trainable_summary with frozen components."""
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Freeze vision encoder to test trainable_only=True path
        backbone.freeze_vision_encoder()
        
        summary = backbone.get_trainable_summary()
        
        # Vision encoder should show as not trainable
        self.assertFalse(summary["vision_encoder"]["trainable"])
        self.assertFalse(summary["vision_projector"]["trainable"])
        
        # Total params should still be counted correctly
        self.assertGreater(summary["total"]["total_params"], 0)
        # Trainable params should be less than total
        self.assertLess(
            summary["total"]["trainable_params"],
            summary["total"]["total_params"]
        )

    def test_print_trainable_summary(self):
        """Test print_trainable_summary executes without error."""
        import io
        import sys
        
        backbone = Llama3VisionBackbone(config=self.config)
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            backbone.print_trainable_summary()
            output = captured_output.getvalue()
            
            # Check that output contains expected strings
            self.assertIn("TRAINABLE PARAMETERS SUMMARY", output)
            self.assertIn("vision_encoder", output)
            self.assertIn("text_backbone", output)
            # Check for cross_attention which uses the elif branch
            self.assertIn("cross_attention", output)
            self.assertIn("layers trainable", output)
        finally:
            sys.stdout = sys.__stdout__
