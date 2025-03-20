from keras_hub.src.models.roformerV2.roformerV2_tokenizer import (
    RoformerV2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class RoformerV2TokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.init_kwargs = {"vocabulary": self.vocab}
        self.input_data = ["THE QUICK BROWN FOX", "THE FOX"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=RoformerV2Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[5, 6, 7, 8], [5, 8]],
        )

    def test_lowercase(self):
        tokenizer = RoformerV2Tokenizer(vocabulary=self.vocab, lowercase=True)
        output = tokenizer(self.input_data)
        self.assertAllEqual(output, [[9, 10, 11, 12], [9, 12]])

    def test_tokenizer_special_tokens(self):
        input_data = ["[CLS] THE [MASK] FOX [SEP] [PAD]"]
        tokenizer = RoformerV2Tokenizer(
            **self.init_kwargs, special_tokens_in_strings=True
        )
        output_data = tokenizer(input_data)
        expected_output = [[2, 5, 4, 8, 3, 0]]

        self.assertAllEqual(output_data, expected_output)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            RoformerV2Tokenizer(vocabulary=["a", "b", "c"])
