from keras_hub.src.models.qwen3_asr.qwen3_asr_tokenizer import Qwen3ASRTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = {
            "<|endoftext|>": 0,
            "<|im_end|>": 1,
            "the": 2,
            "Ġquick": 3,
            "Ġbrown": 4,
            "Ġfox": 5,
        }
        self.merges = ["t h", "th e", "q u", "qu ick"]
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
        }

    def test_tokenize_known_token(self):
        tokenizer = Qwen3ASRTokenizer(**self.init_kwargs)
        output = tokenizer(["the"])
        # "the" should map to token ID 2.
        self.assertEqual(output[0][0], 2)

    def test_special_tokens(self):
        tokenizer = Qwen3ASRTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.end_token_id, 1)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertIsNone(tokenizer.start_token_id)

    def test_serialization(self):
        tokenizer = Qwen3ASRTokenizer(**self.init_kwargs)
        config = tokenizer.get_config()
        restored = Qwen3ASRTokenizer.from_config(config)
        self.assertEqual(restored.get_config(), config)
