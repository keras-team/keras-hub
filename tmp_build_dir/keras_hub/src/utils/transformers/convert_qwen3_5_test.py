import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen3_5


class TestQwen3_5Converter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Qwen3_5Backbone.from_preset(
            "hf://yujiepan/qwen3.5-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3_5Backbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/qwen3.5-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3_5Backbone)

    def test_qwen3_5_rope_theta(self):
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
            "partial_rotary_factor": 0.5,
            "rope_parameters": {"partial_rotary_factor": 0.5},
            "tie_word_embeddings": True,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
        }
        keras_config = convert_qwen3_5.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {
            "rope_theta": 20000.0,
            "partial_rotary_factor": 0.5,
        }
        keras_config = convert_qwen3_5.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
