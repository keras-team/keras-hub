import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.models.llama3.llama3_causal_lm import Llama3CausalLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
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

    # TODO: compare numerics with huggingface model
