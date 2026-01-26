import keras
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import Qwen3OmniBackbone
from keras_hub.src.models.qwen3_omni.qwen3_omni_causal_lm import Qwen3OmniCausalLM
from keras_hub.src.tests.test_case import TestCase


# NOTE: This test is valid and should pass locally. It is skipped only on
# TensorFlow GPU CI because of ResourceExhaustedError (OOM). Revisit once
# TensorFlow GPU CI runs without hitting OOM.
@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
)
class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = Qwen3OmniCausalLM.from_preset("hf://Qwen/Qwen3-Omni-30B-A3B-Instruct")
        prompt = "What is Keras?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "hf://Qwen/Qwen3-Omni-30B-A3B-Instruct"
        model = CausalLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3OmniCausalLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3OmniBackbone)
