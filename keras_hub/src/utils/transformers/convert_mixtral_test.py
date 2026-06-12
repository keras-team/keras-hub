import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.mixtral.mixtral_backbone import MixtralBackbone
from keras_hub.src.models.mixtral.mixtral_causal_lm import MixtralCausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_mixtral


class TestMixtralConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = MixtralCausalLM.from_preset("hf://mistralai/Mixtral-8x7B-v0.1")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://mistralai/Mixtral-8x7B-v0.1",
            load_weights=False,
        )
        self.assertIsInstance(model, MixtralCausalLM)
        model = Backbone.from_preset(
            "hf://mistralai/Mixtral-8x7B-v0.1",
            load_weights=False,
        )
        self.assertIsInstance(model, MixtralBackbone)

    def test_mixtral_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_key_value_heads": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
            "output_router_logits": False,
        }
        keras_config = convert_mixtral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config["rope_parameters"] = {"rope_theta": 20000.0}
        keras_config = convert_mixtral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)
