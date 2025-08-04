import pytest

from keras_hub.src.models.falcon.falcon_backbone import FalconBackbone
from keras_hub.src.models.falcon.falcon_causal_lm import FalconCausalLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = FalconCausalLM.from_preset("tiiuae/falcon-rw-1b")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = FalconCausalLM.from_preset("tiiuae/falcon-rw-1b")
        self.assertIsInstance(model, FalconCausalLM)
        model = FalconBackbone.from_preset(
            "hf://tiiuae/falcon-1b",
            load_weights=False,
        )
        self.assertIsInstance(model, FalconBackbone)
