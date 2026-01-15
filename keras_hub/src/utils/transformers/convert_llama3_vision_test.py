"""Tests for Llama 3.2 Vision weight conversion."""

import pytest

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_llama3_vision


class ConvertLlama3VisionTest(TestCase):
    @pytest.mark.large
    def test_convert_backbone_config(self):
        """Test config conversion from HuggingFace format."""
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
            },
            "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38],
        }

        config = convert_llama3_vision.convert_backbone_config(hf_config)

        # Verify flattened config
        self.assertEqual(config["vocabulary_size"], 128256)
        self.assertEqual(config["num_layers"], 40)
        self.assertEqual(config["hidden_dim"], 4096)
        self.assertEqual(config["vision_hidden_dim"], 1280)
        self.assertEqual(config["vision_num_layers"], 32)
        self.assertEqual(config["vision_patch_size"], 14)
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

        config = convert_llama3_vision.load_image_converter_config(
            "nonexistent_preset", hf_config
        )
        self.assertEqual(config["image_size"], 560)

    def test_config_without_vision(self):
        """Test configs without vision return None for image converter."""
        hf_config = {"text_config": {"vocab_size": 128256}}

        config = convert_llama3_vision.load_image_converter_config(
            "preset", hf_config
        )
        self.assertIsNone(config)
