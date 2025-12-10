import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Gemma3CausalLM.from_preset("google/gemma-3-1b-it")
        prompt = "What is the capital of India?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "google/gemma-3-1b-it"
        model = CausalLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma3CausalLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma3Backbone)
