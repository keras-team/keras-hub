import os
import pytest
from keras_hub.src.models.deepseek_v31.deepseek_v31_tokenizer import (
    DeepSeekV31Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class DeepSeekV31TokenizerTest(TestCase):
    def setUp(self):
        self.vocab = {
            "<｜begin▁of▁sentence｜>": 151646,
            "<｜end▁of▁sentence｜>": 151643,
        }
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ Ġ"):
            self.vocab[c] = i + 2

        # Register the fully formed BPE chunks that get merged to prevent dropping them during detokenize
        self.vocab["th"] = 100
        self.vocab["ea"] = 101

        self.merges = ["t h", "e a"]

        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=DeepSeekV31Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_special_tokens(self):
        tokenizer = DeepSeekV31Tokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.start_token_id, 151646)
        self.assertEqual(tokenizer.end_token_id, 151643)

    def test_tokenizer_vocab_size(self):
        tokenizer = DeepSeekV31Tokenizer(**self.init_kwargs)
        self.assertGreater(tokenizer.vocabulary_size(), 0)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DeepSeekV31Tokenizer.presets:
            self.run_preset_test(
                cls=DeepSeekV31Tokenizer,
                preset=preset,
                input_data=["the quick brown fox"],
            )
