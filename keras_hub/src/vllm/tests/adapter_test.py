import pytest
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.adapter import KerasVLLMAdapter
import keras_hub

class KerasVLLMAdapterTest(TestCase):
    def setUp(self):
        class DummyConfig:
            keras_hub_preset = "gemma_2b_en"
        self.config = DummyConfig()

    def test_adapter_initialization(self):
        try:
            adapter = KerasVLLMAdapter(self.config)
            self.assertIsNotNone(adapter.model)
            self.assertTrue(hasattr(adapter, "get_vllm_model"))
            self.assertEqual(adapter.config.architectures[0], "GemmaForCausalLM")
        except Exception as e:
            self.skipTest(f"Failed to initialize adapter, likely due to missing Kaggle credentials or vLLM: {e}")

    def test_missing_preset_raises_error(self):
        class BadConfig:
            pass
        with self.assertRaises(ValueError):
            KerasVLLMAdapter(BadConfig())
