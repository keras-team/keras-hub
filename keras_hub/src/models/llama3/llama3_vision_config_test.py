"""Tests for Llama3VisionConfig and Llama3VisionEncoderConfig."""

from keras_hub.src.models.llama3.llama3_vision_config import Llama3VisionConfig
from keras_hub.src.models.llama3.llama3_vision_config import (
    Llama3VisionEncoderConfig,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionEncoderConfigTest(TestCase):
    """Test cases for Llama3VisionEncoderConfig."""

    def test_is_two_stage_property_single_stage(self):
        """Test is_two_stage returns False for single-stage config."""
        config = Llama3VisionEncoderConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
        )
        self.assertFalse(config.is_two_stage)

    def test_is_two_stage_property_two_stage(self):
        """Test is_two_stage returns True for two-stage config."""
        config = Llama3VisionEncoderConfig(
            hidden_dim=256,
            num_layers=6,
            num_heads=4,
            local_layers=4,
            global_layers=2,
        )
        self.assertTrue(config.is_two_stage)

    def test_get_config_single_stage(self):
        """Test get_config for single-stage encoder."""
        config = Llama3VisionEncoderConfig(
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            intermediate_dim=2048,
            patch_size=16,
            image_size=224,
        )
        config_dict = config.get_config()

        self.assertEqual(config_dict["hidden_dim"], 512)
        self.assertEqual(config_dict["num_layers"], 8)
        self.assertEqual(config_dict["num_heads"], 8)
        self.assertEqual(config_dict["intermediate_dim"], 2048)
        self.assertEqual(config_dict["patch_size"], 16)
        self.assertEqual(config_dict["image_size"], 224)
        self.assertIsNone(config_dict["local_layers"])
        self.assertIsNone(config_dict["global_layers"])

    def test_get_config_two_stage(self):
        """Test get_config for two-stage encoder."""
        config = Llama3VisionEncoderConfig(
            hidden_dim=256,
            num_layers=6,
            num_heads=4,
            local_layers=4,
            global_layers=2,
        )
        config_dict = config.get_config()

        self.assertEqual(config_dict["local_layers"], 4)
        self.assertEqual(config_dict["global_layers"], 2)
        self.assertTrue(config.is_two_stage)


class Llama3VisionConfigTest(TestCase):
    """Test cases for Llama3VisionConfig."""

    def test_default_vision_encoder_config(self):
        """Test that None vision_encoder_config creates default."""
        config = Llama3VisionConfig(
            vision_encoder_config=None,
            text_config={"hidden_dim": 256},
        )

        self.assertIsInstance(
            config.vision_encoder_config, Llama3VisionEncoderConfig
        )
        # Check default values
        self.assertEqual(config.vision_encoder_config.hidden_dim, 1152)

    def test_dict_vision_encoder_config(self):
        """Test initialization with dict vision_encoder_config."""
        vision_config_dict = {
            "hidden_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "patch_size": 14,
        }
        config = Llama3VisionConfig(
            vision_encoder_config=vision_config_dict,
            text_config={"hidden_dim": 256},
        )

        self.assertIsInstance(
            config.vision_encoder_config, Llama3VisionEncoderConfig
        )
        self.assertEqual(config.vision_encoder_config.hidden_dim, 512)
        self.assertEqual(config.vision_encoder_config.num_layers, 8)

    def test_default_text_config(self):
        """Test that None text_config creates empty dict."""
        config = Llama3VisionConfig(
            vision_encoder_config=Llama3VisionEncoderConfig(),
            text_config=None,
        )

        self.assertIsInstance(config.text_config, dict)
        self.assertEqual(config.text_config, {})

    def test_get_config_with_dict_text_config(self):
        """Test get_config with dict text_config."""
        text_config_dict = {
            "hidden_dim": 2048,
            "num_layers": 24,
        }
        config = Llama3VisionConfig(
            vision_encoder_config=Llama3VisionEncoderConfig(hidden_dim=256),
            text_config=text_config_dict,
            cross_attention_layers=[3, 8],
        )

        config_dict = config.get_config()

        self.assertIn("vision_encoder_config", config_dict)
        self.assertEqual(config_dict["text_config"], text_config_dict)
        self.assertEqual(config_dict["cross_attention_layers"], [3, 8])

    def test_get_config_with_object_text_config(self):
        """Test get_config when text_config has get_config method."""

        # Create a mock config object with get_config
        class MockTextConfig:
            def get_config(self):
                return {"hidden_dim": 1024, "num_layers": 16}

        text_config = MockTextConfig()
        config = Llama3VisionConfig(
            vision_encoder_config=Llama3VisionEncoderConfig(),
            text_config=text_config,
        )

        config_dict = config.get_config()

        self.assertEqual(
            config_dict["text_config"], {"hidden_dim": 1024, "num_layers": 16}
        )

    def test_default_cross_attention_layers(self):
        """Test default cross_attention_layers."""
        config = Llama3VisionConfig(
            vision_encoder_config=Llama3VisionEncoderConfig(),
            text_config={},
        )

        # Should use default layers
        self.assertEqual(
            config.cross_attention_layers, [3, 8, 13, 18, 23, 28, 33, 38]
        )

    def test_custom_cross_attention_layers(self):
        """Test custom cross_attention_layers."""
        custom_layers = [1, 5, 10]
        config = Llama3VisionConfig(
            vision_encoder_config=Llama3VisionEncoderConfig(),
            text_config={},
            cross_attention_layers=custom_layers,
        )

        self.assertEqual(config.cross_attention_layers, custom_layers)
