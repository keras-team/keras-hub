import unittest

import keras_hub


class ImportTest(unittest.TestCase):
    def test_version(self):
        self.assertIsNotNone(keras_hub.__version__)
        
    def test_qwen3_tts_is_registered(self):
        self.assertTrue(
            hasattr(keras_hub.models, "Qwen3TTS"),
            "Qwen3TTS model should be registered in keras_hub.models",
        )
        
