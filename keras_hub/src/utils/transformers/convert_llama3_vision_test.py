"""Tests for Llama 3.2 Vision weight conversion."""

import pytest

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_llama3_vision


class ConvertLlama3VisionTest(TestCase):
    """Test cases for Llama 3.2 Vision weight conversion."""

    @pytest.mark.large
    def test_convert_backbone_config(self):
        """Test config conversion from HuggingFace format."""
        # Mock HuggingFace config (simplified)
        hf_config = {
            "model_type": "mllama",
            "vision_config": {
                "hidden_size": 1280,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "intermediate_size": 5120,
                "patch_size": 14,
                "image_size": 560,
                "num_channels": 3,
                "layer_norm_eps": 1e-6,
            },
            "text_config": {
                "vocab_size": 128256,
                "num_hidden_layers": 40,
                "num_attention_heads": 32,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_key_value_heads": 8,
                "rope_theta": 500000,
                "rms_norm_eps": 1e-5,
                "tie_word_embeddings": False,
            },
            "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38],
        }

        config = convert_llama3_vision.convert_backbone_config(hf_config)

        # Verify vision config
        self.assertEqual(config["vision_encoder_config"]["hidden_dim"], 1280)
        self.assertEqual(config["vision_encoder_config"]["num_layers"], 32)
        self.assertEqual(config["vision_encoder_config"]["patch_size"], 14)

        # Verify text config
        self.assertEqual(config["text_config"]["vocabulary_size"], 128256)
        self.assertEqual(config["text_config"]["num_layers"], 40)
        self.assertEqual(config["text_config"]["hidden_dim"], 4096)

        # Verify cross-attention layers
        self.assertEqual(
            config["cross_attention_layers"], [3, 8, 13, 18, 23, 28, 33, 38]
        )

    def test_load_image_converter_config(self):
        """Test image converter config loading."""
        hf_config = {
            "vision_config": {
                "image_size": 560,
            }
        }

        # Test fallback when preprocessor_config.json is not available
        config = convert_llama3_vision.load_image_converter_config(
            "nonexistent_preset", hf_config
        )

        # Should return default values
        self.assertEqual(config["image_size"], 560)

    def test_config_without_vision(self):
        """Test that configs without vision return None for image converter."""
        hf_config = {"text_config": {"vocab_size": 128256}}

        config = convert_llama3_vision.load_image_converter_config(
            "preset", hf_config
        )
        self.assertIsNone(config)
