import keras
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.smollm3.smollm3_backbone import SmolLM3Backbone
from keras_hub.src.models.smollm3.smollm3_causal_lm import SmolLM3CausalLM
from keras_hub.src.tests.test_case import TestCase


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
)
class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = SmolLM3CausalLM.from_preset("hf://HuggingFaceTB/SmolLM3-3B")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://HuggingFaceTB/SmolLM3-3B",
            load_weights=False,
        )
        self.assertIsInstance(model, SmolLM3CausalLM)
        model = Backbone.from_preset(
            "hf://HuggingFaceTB/SmolLM3-3B",
            load_weights=False,
        )
        self.assertIsInstance(model, SmolLM3Backbone)
