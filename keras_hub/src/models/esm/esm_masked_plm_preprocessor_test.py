from keras_hub.src.models.esm.esm_masked_plm_preprocessor import (
    ESMMaskedPLMPreprocessor,
)
from keras_hub.src.models.esm.esm_tokenizer import (
    ESMTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ESMMaskedPLMPreprocessort(TestCase):
    def setUp(self):
        self.vocab = [ "[UNK]", "[PAD]","[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["THE", "QUICK", "BROWN", "FOX"]
        self.vocab += ["the", "quick", "brown", "fox"]
        self.tokenizer = ESMTokenizer(vocabulary=self.vocab)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            # Simplify our testing by masking every available token.
            "mask_selection_rate": 1.0,
            "mask_token_rate": 1.0,
            "random_token_rate": 0.0,
            "mask_selection_length": 4,
            "sequence_length": 12,
        }
        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=ESMMaskedPLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[2, 4, 4, 4, 4, 3, 1, 1, 1, 1, 1, 1]],
                    "mask_positions": [[1, 2, 3, 4]],
                },
                [[9, 10, 11, 12]],
                [[1.0, 1.0, 1.0, 1.0]],
            ),
        )

    def test_no_masking_zero_rate(self):
        no_mask_preprocessor = ESMMaskedPLMPreprocessor(
            self.tokenizer,
            mask_selection_rate=0.0,
            mask_selection_length=4,
            sequence_length=12,
        )
        input_data = ["the quick brown fox"]
        self.assertAllClose(
            no_mask_preprocessor(input_data),
            (
                {
                    "token_ids": [[2, 9, 10, 11, 12, 3, 1, 1, 1, 1, 1, 1]],
                    "mask_positions": [[0, 0, 0, 0]],
                },
                [[0, 0, 0, 0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ),
        )
