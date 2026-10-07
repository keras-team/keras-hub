import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
    Gemma4AssistantCausalLM,
)
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.tests.test_case import TestCase


class TestGemma4AssistantConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        # `Gemma4AssistantCausalLM` is a speculative-decoding draft model —
        # it has no preprocessor and must never be called via `generate()`
        # directly (see its docstring). This just verifies the preset's
        # weights convert without error.
        model = Gemma4AssistantCausalLM.from_preset(
            "hf://yujiepan/gemma-4-assistant-tiny-random"
        )
        self.assertIsInstance(model, Gemma4AssistantCausalLM)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "hf://yujiepan/gemma-4-assistant-tiny-random"
        model = CausalLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma4AssistantCausalLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, Gemma4Backbone)
