from keras_hub.src.models.esm.esm_tokenizer import (
    ESMTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ESMTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = [ "[UNK]", "[PAD]","[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.init_kwargs = {"vocabulary": self.vocab}
        self.input_data = ["THE QUICK BROWN FOX", "THE FOX"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=ESMTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[5, 6, 7, 8], [5, 8]],
        )

    def test_lowercase(self):
        tokenizer = ESMTokenizer(vocabulary=self.vocab, lowercase=True)
        output = tokenizer(self.input_data)
        self.assertAllEqual(output, [[9, 10, 11, 12], [9, 12]])

    def test_tokenizer_special_tokens(self):
        input_data = ["[CLS] THE [MASK] FOX [SEP] [PAD]"]
        tokenizer = ESMTokenizer(
            **self.init_kwargs, special_tokens_in_strings=True
        )
        output_data = tokenizer(input_data)
        expected_output = [[2, 5, 4, 8, 3, 1]]

        self.assertAllEqual(output_data, expected_output)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            ESMTokenizer(vocabulary=["a", "b", "c"])
