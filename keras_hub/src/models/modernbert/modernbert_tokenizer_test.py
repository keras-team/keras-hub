from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTokenizerTest(TestCase):
    """
    Tests for verifying the `ModernBertTokenizer`
    implementation details.
    """

    def setUp(self):
        self.vocab = [
            "<|endoftext|>",
            "<|padding|>",
            "<mask>",
            "air",
            "Ġair",
            "plane",
            "Ġat",
        ]
        self.vocab += ["port", "[UNK]"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "[CLS] airplane at airport[SEP][PAD]",
            " airplane airport",
        ]

    def test_errors_missing_special_tokens(self):
        """
        Verify that initialization fails gracefully
        when special tokens are missing.
        """
        with self.assertRaises(ValueError):
            ModernBertTokenizer(vocabulary=["a", "b", "c"], merges=[])
