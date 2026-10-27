import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen.qwen_backbone import QwenBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen


class TestQwenConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = QwenBackbone.from_preset(
            "hf://yujiepan/qwen2-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, QwenBackbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/qwen2-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, QwenBackbone)

    def test_qwen_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "use_sliding_window": False,
            "sliding_window": 4096,
            "tie_word_embeddings": True,
        }
        keras_config = convert_qwen.convert_backbone_config(transformers_config)
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_qwen.convert_backbone_config(transformers_config)
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
