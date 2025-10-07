from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer
from keras_hub.src.tests.test_case import TestCase


class RWKV7TokenizerTest(TestCase):
    def setUp(self):
        self.tokenizer = RWKVTokenizer(
            ["1 ' ' 1", "2 '\\n' 1", "3 'the' 3", "4 'hello' 5", "5 'world' 5"]
        )

    def test_tokenizer_basics(self):
        result = self.tokenizer("hello world")
        self.assertAllEqual(result, [[4, 1, 5]])

    def test_vocabulary_size(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 5)

    def test_tokenize_and_detokenize(self):
        # Test detokenization
        text = self.tokenizer.detokenize([[4, 1, 5]])
        self.assertEqual(text[0], "hello world")

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.end_token_id, 2)
