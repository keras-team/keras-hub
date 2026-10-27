import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_llama3


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Llama3CausalLM.from_preset("hf://ariG23498/tiny-llama3-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://ariG23498/tiny-llama3-test",
            load_weights=False,
        )
        self.assertIsInstance(model, Llama3CausalLM)
        model = Backbone.from_preset(
            "hf://ariG23498/tiny-llama3-test",
            load_weights=False,
        )
        self.assertIsInstance(model, Llama3Backbone)

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
            "tie_word_embeddings": True,
        }
        keras_config = convert_llama3.convert_backbone_config(
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
            "tie_word_embeddings": True,
        }
        keras_config = convert_llama3.convert_backbone_config(
            transformers_config
        )
        self.assertEqual(keras_config["rope_max_wavelength"], 20000.0)

    # TODO: compare numerics with huggingface model
