import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_gemma


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = GemmaCausalLM.from_preset("hf://ariG23498/tiny-gemma-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

        model = GemmaCausalLM.from_preset("hf://ariG23498/tiny-gemma-2-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://ariG23498/tiny-gemma-test",
            load_weights=False,
        )
        self.assertIsInstance(model, GemmaCausalLM)
        model = Backbone.from_preset(
            "hf://ariG23498/tiny-gemma-test",
            load_weights=False,
        )
        self.assertIsInstance(model, GemmaBackbone)
        model = Backbone.from_preset(
            "hf://hf-tiny-v2/tiny-random-VaultGemmaForCausalLM",
            load_weights=False,
        )
        self.assertIsInstance(model, GemmaBackbone)

    def test_convert_vaultgemma_backbone_config(self):
        hf_config = {
            "model_type": "vaultgemma",
            "vocab_size": 256000,
            "num_hidden_layers": 26,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_size": 1152,
            "intermediate_size": 6912,
            "head_dim": 256,
            "query_pre_attn_scalar": 256,
            "sliding_window": 512,
        }
        backbone_config = convert_gemma.convert_backbone_config(hf_config)
        self.assertEqual(backbone_config["vocabulary_size"], 256000)
        self.assertEqual(backbone_config["num_layers"], 26)
        self.assertEqual(backbone_config["use_post_ffw_norm"], False)
        self.assertEqual(backbone_config["use_post_attention_norm"], False)
        self.assertEqual(backbone_config["query_head_dim_normalize"], True)
        self.assertEqual(backbone_config["sliding_window_size"], 512)

    # TODO: compare numerics with huggingface model
