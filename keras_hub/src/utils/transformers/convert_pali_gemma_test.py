import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_hub.src.models.pali_gemma.pali_gemma_causal_lm import (
    PaliGemmaCausalLM,
)
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = PaliGemmaCausalLM.from_preset(
            "hf://ariG23498/tiny-pali-gemma-test"
        )
        image = np.random.rand(224, 224, 3)
        prompt = "describe the image "
        model.generate({"images": image, "prompts": prompt}, max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = CausalLM.from_preset(
            "hf://ariG23498/tiny-pali-gemma-test",
            load_weights=False,
        )
        self.assertIsInstance(model, PaliGemmaCausalLM)
        model = Backbone.from_preset(
            "hf://ariG23498/tiny-pali-gemma-test",
            load_weights=False,
        )
        self.assertIsInstance(model, PaliGemmaBackbone)

    # TODO: compare numerics with huggingface model
