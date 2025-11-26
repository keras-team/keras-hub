from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer
from keras_hub.src.tests.test_case import TestCase


class RWKVTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = [
            "0 '\\n' 1",
            "1 ' ' 1",
            "2 'the' 3",
            "3 'def' 3",
            "4 'code' 4",
            "5 'hello' 5",
            "6 'world' 5",
            "7 'python' 6",
            "8 'return' 6",
            "9 'function' 8",
        ]
        self.tokenizer = RWKVTokenizer(
            vocabulary=self.vocab,
            pad_token_id=0,
        )
        self.input_data = [
            "hello world",
            "def function",
            "the python",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=RWKVTokenizer,
            init_kwargs={"vocabulary": self.vocab, "pad_token_id": 0},
            input_data=self.input_data,
            expected_output=[[5, 1, 6], [3, 1, 9], [2, 1, 7]],
        )

    def test_tokenize_basics(self):
        result = self.tokenizer("hello world")
        self.assertAllEqual(result, [5, 1, 6])

    def test_tokenize_list(self):
        result = self.tokenizer(["hello world", "the python code"])
        self.assertAllEqual(result, [[5, 1, 6], [2, 1, 7, 1, 4]])

    def test_detokenize(self):
        result = self.tokenizer.detokenize([[5, 1, 6, 0, 0]])
        self.assertAllEqual(result, ["hello world"])

    def test_detokenize_batch(self):
        result = self.tokenizer.detokenize([[5, 1, 6], [3, 1, 9, 0]])
        self.assertAllEqual(result, ["hello world", "def function"])

    def test_accessors(self):
        self.assertEqual(self.tokenizer.vocabulary_size(), 10)
        self.assertEqual(self.tokenizer.get_vocabulary(), self.vocab)
        self.assertEqual(self.tokenizer.id_to_token(5), b"hello")
        self.assertEqual(self.tokenizer.token_to_id(b"world"), 6)
        self.assertEqual(self.tokenizer.pad_token_id, 0)

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
        self.run_preprocessing_layer_test(
            cls=RWKVTokenizer,
            init_kwargs={"vocabulary": self.vocab, "pad_token_id": 0},
            input_data=["hello world", "def function", "pythoncodereturn"],
            expected_output=[[5, 1, 6], [3, 1, 9], [7, 4, 8]],
        )
