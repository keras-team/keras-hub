import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm import GemmaCausalLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
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

    # TODO: compare numerics with huggingface model
