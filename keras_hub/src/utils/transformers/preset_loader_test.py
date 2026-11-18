from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_gemma4_assistant
from keras_hub.src.utils.transformers import convert_gemma4_unified
from keras_hub.src.utils.transformers.preset_loader import (
    TransformersPresetLoader,
)


class TransformersPresetLoaderTest(TestCase):
    def test_gemma4_unified_assistant_routing(self):
        config_assistant = {
            "model_type": "gemma4_unified_assistant",
            "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
        }
        config_unified = {
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForCausalLM"],
        }

        loader_assistant = TransformersPresetLoader(
            "hf://dummy", config_assistant
        )
        self.assertIs(loader_assistant.converter, convert_gemma4_assistant)

        loader_unified = TransformersPresetLoader("hf://dummy", config_unified)
        self.assertIs(loader_unified.converter, convert_gemma4_unified)
