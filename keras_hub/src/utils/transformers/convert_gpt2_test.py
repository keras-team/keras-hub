import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = GPT2CausalLM.from_preset("hf://openai-community/gpt2")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://openai-community/gpt2",
            load_weights=False,
        )
        self.assertIsInstance(model, GPT2CausalLM)
        model = Backbone.from_preset(
            "hf://openai-community/gpt2",
            load_weights=False,
        )
        self.assertIsInstance(model, GPT2Backbone)

    # TODO: compare numerics with huggingface model
