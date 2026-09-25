import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bloom.bloom_backbone import BloomBackbone
from keras_hub.src.models.bloom.bloom_causal_lm import BloomCausalLM
from keras_hub.src.models.bloom.bloom_tokenizer import BloomTokenizer
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_bloom


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = BloomCausalLM.from_preset("hf://bigscience/bloom-560m")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://bigscience/bloom-560m",
            load_weights=False,
        )
        self.assertIsInstance(model, BloomCausalLM)
        model = Backbone.from_preset(
            "hf://bigscience/bloom-560m",
            load_weights=False,
        )
        self.assertIsInstance(model, BloomBackbone)

    @pytest.mark.large
    def test_convert_tokenizer(self):
        tokenizer = BloomTokenizer.from_preset("hf://bigscience/bloom-560m")
        self.assertIsInstance(tokenizer, BloomTokenizer)
        self.assertEqual(
            tokenizer.detokenize(tokenizer("The quick brown fox")),
            "The quick brown fox",
        )

    def test_convert_backbone_config(self):
        hf_config = {
            "vocab_size": 250880,
            "n_layer": 24,
            "n_head": 16,
            "hidden_size": 1024,
            "hidden_dropout": 0.1,
            "layer_norm_epsilon": 1e-5,
        }
        config = convert_bloom.convert_backbone_config(hf_config)
        self.assertEqual(config["vocabulary_size"], 250880)
        self.assertEqual(config["num_layers"], 24)
        self.assertEqual(config["num_heads"], 16)
        self.assertEqual(config["hidden_dim"], 1024)
        self.assertEqual(config["intermediate_dim"], 4096)
        self.assertEqual(config["dropout"], 0.1)
        self.assertEqual(config["layer_norm_epsilon"], 1e-5)

    def test_convert_legacy_backbone_config(self):
        # Checkpoints such as `bigscience/bloom-560m` use the older key
        # spellings `n_embed` and `num_attention_heads`.
        hf_config = {
            "vocab_size": 250880,
            "n_layer": 24,
            "num_attention_heads": 16,
            "n_embed": 1024,
            "n_inner": None,
            "hidden_dropout": 0.0,
            "layer_norm_epsilon": 1e-5,
        }
        config = convert_bloom.convert_backbone_config(hf_config)
        self.assertEqual(config["num_heads"], 16)
        self.assertEqual(config["hidden_dim"], 1024)
        self.assertEqual(config["intermediate_dim"], 4096)

    # TODO: compare numerics with huggingface model
