import pytest
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm.tokenizer import KerasVLLMTokenizerAdapter
import keras_hub

class KerasVLLMTokenizerAdapterTest(TestCase):
    def setUp(self):
        try:
            self.tokenizer = keras_hub.models.GemmaTokenizer.from_preset("gemma_2b_en")
            self.adapter = KerasVLLMTokenizerAdapter(self.tokenizer)
        except Exception as e:
            self.skipTest(f"Failed to initialize tokenizer, likely due to Kaggle credentials: {e}")

    def test_vocab_size(self):
        self.assertGreater(self.adapter.vocab_size, 0)
        
    def test_get_vocab(self):
        vocab = self.adapter.get_vocab()
        self.assertIsInstance(vocab, dict)
        self.assertTrue(len(vocab) > 0)
        
    def test_encode_decode(self):
        text = "Hello Keras Hub"
        token_ids = self.adapter.encode(text)
        self.assertIsInstance(token_ids, list)
        self.assertTrue(len(token_ids) > 0)
        
        decoded = self.adapter.decode(token_ids)
        self.assertIn("Keras Hub", decoded)
