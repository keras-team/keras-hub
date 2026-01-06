import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.models.gpt_oss.gpt_oss_causal_lm import GptOssCausalLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = GptOssCausalLM.from_preset("gpt_oss_20b_en")
        prompt = "What is the capital of India?"
        model.generate([prompt], max_length=15)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "gpt_oss_20b_en"
        model = CausalLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, GptOssCausalLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, GptOssBackbone)
