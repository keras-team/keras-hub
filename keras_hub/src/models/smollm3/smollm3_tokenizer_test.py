from keras_hub.src.models.smollm3.smollm3_tokenizer import SmolLM3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class SmolLM3TokenizerTest(TestCase):
    def test_pre_tokenization(self):
        # SmolLM3 uses the Llama 3 pre-tokenizer regex, which groups digits
        # in runs of up to three. Merges must not cross those boundaries.
        vocab = ["<|begin_of_text|>", "<|end_of_text|>", "<think>"]
        vocab += ["</think>", "Ġ", "2", "0", "4", "20", "24", "202"]
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        merges = ["2 0", "2 4", "20 2"]
        tokenizer = SmolLM3Tokenizer(
            vocabulary=vocab, merges=merges, dtype="string"
        )
        self.assertAllEqual(tokenizer("2024"), ["202", "4"])
        self.assertAllEqual(tokenizer(" 2024"), ["Ġ", "202", "4"])
