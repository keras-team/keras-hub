from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["[CLS]", "[PAD]", "[SEP]", "air", "Ġair", "plane", "Ġat"]
        self.vocab += ["port", "[MASK]", "[UNK]"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "[CLS] airplane at airport[SEP][PAD]",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=ModernBertTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[0, 4, 5, 6, 4, 7, 2, 1], [4, 5, 4, 7]],
            expected_detokenize_output=[
                "[CLS] airplane at airport[SEP][PAD]",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            ModernBertTokenizer(vocabulary=["a", "b", "c"], merges=[])
