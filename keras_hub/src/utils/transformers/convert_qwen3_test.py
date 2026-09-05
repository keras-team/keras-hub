import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3.qwen3_backbone import Qwen3Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen3


class TestQwen3Converter(TestCase):
    @pytest.mark.extra_large
    def test_backbone_from_hf_preset(self):
        model = Qwen3Backbone.from_preset(
            "hf://microsoft/harrier-oss-v1-0.6b",
            load_weights=False,
        )
        # harrier: hidden_dim=1024, num_layers=28
        self.assertEqual(model.hidden_dim, 1024)
        self.assertEqual(model.num_layers, 28)

    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Qwen3Backbone.from_preset(
            "hf://yujiepan/qwen3-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3Backbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/qwen3-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3Backbone)

    def test_qwen3_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "hidden_size": 32,
            "head_dim": 8,
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
        keras_config = convert_qwen3.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_qwen3.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
