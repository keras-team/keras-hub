from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer
from keras_hub.src.tests.test_case import TestCase


class RWKVTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = [
            "0 '\\n' 1",
            "1 ' ' 1",
            "2 'the' 3",
            "3 'hello' 5",
            "4 'world' 5",
        ]
        self.tokenizer = RWKVTokenizer(vocabulary=self.vocab)

    def test_tokenize_basics(self):
        result = self.tokenizer("hello world")
        self.assertAllEqual(result, [3, 1, 4])

    def test_tokenize_list(self):
        result = self.tokenizer(["hello world", "the world world"])
        self.assertAllEqual(result, [[3, 1, 4], [2, 1, 4, 1, 4]])

    def test_detokenize(self):
        result = self.tokenizer.detokenize([[3, 1, 4, 0]])
        self.assertAllEqual(result, ["hello world"])

    def test_detokenize_batch(self):
        result = self.tokenizer.detokenize([[3, 1, 4], [2, 1, 4, 0]])
        self.assertAllEqual(result, ["hello world", "the world"])

    def test_accessors(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 5)
        self.assertEqual(self.tokenizer.get_vocabulary(), self.vocab)
        self.assertEqual(self.tokenizer.id_to_token(3), b"hello")
        self.assertEqual(self.tokenizer.token_to_id(b"world"), 4)

    def test_error_id_out_of_vocabulary(self):
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(100)
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(-1)

    def test_config(self):
        config = self.tokenizer.get_config()
        cloned = RWKVTokenizer.from_config(config)
        cloned.set_vocabulary(self.vocab)

        inputs = ["hello world"]
        self.assertAllEqual(self.tokenizer(inputs), cloned(inputs))

    def test_preprocessing_layer(self):
        """Standard preprocessing layer test."""
        self.run_preprocessing_layer_test(
            cls=RWKVTokenizer,
            init_kwargs={"vocabulary": self.vocab},
            input_data=["hello world", "the world"],
            expected_output=[[3, 1, 4], [2, 1, 4]],
        )
