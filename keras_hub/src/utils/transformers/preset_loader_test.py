from keras_hub.src.tests.test_case import TestCase


class TransformersPresetLoaderTest(TestCase):
    def test_gemma4_unified_assistant_routing(self):
        config = {
            "model_type": "gemma4_unified_assistant",
            "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
        }
        # Test convert_gemma4_assistant check_backbone_class
        from keras_hub.src.utils.transformers import convert_gemma4_assistant

        cls = convert_gemma4_assistant.check_backbone_class(config)
        self.assertEqual(cls.__name__, "Gemma4UnifiedBackbone")
