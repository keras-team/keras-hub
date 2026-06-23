import pytest
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.registry import register_keras_hub_models

class RegistryTest(TestCase):
    def test_register_keras_hub_models(self):
        try:
            import vllm
            register_keras_hub_models()
            from vllm.model_executor.models import ModelRegistry
            self.assertTrue("KerasVLLMAdapter" in ModelRegistry.get_supported_archs())
        except ImportError:
            self.skipTest("vLLM not installed. Skipping registry tests.")
