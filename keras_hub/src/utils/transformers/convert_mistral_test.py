import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_mistral


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = MistralCausalLM.from_preset("hf://cosmo3769/tiny-mistral-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://cosmo3769/tiny-mistral-test",
            load_weights=False,
        )
        self.assertIsInstance(model, MistralCausalLM)
        model = Backbone.from_preset(
            "hf://cosmo3769/tiny-mistral-test",
            load_weights=False,
        )
        self.assertIsInstance(model, MistralBackbone)

    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        transformers_config = {
            "vocab_size": 100,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_key_value_heads": 2,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
        }
        keras_config = convert_mistral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)

        # transformers >= 5 format
        transformers_config = {
            "vocab_size": 100,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_theta": 20000.0},
            "rms_norm_eps": 1e-5,
            "sliding_window": 4096,
        }
        # In the real transformers >= 5, rope_theta might still be present at
        # top level for some models, but the source of truth moved to
        # rope_parameters.
        keras_config = convert_mistral.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)

    # TODO: compare numerics with huggingface model
