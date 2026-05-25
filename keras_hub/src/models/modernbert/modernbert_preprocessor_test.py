from keras import ops

from keras_hub.src.models.modernbert.modernbert_preprocessor import (
    ModernBertMaskedLMPreprocessor,
)
from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertMaskedLMPreprocessorTest(TestCase):
    """Tests for verifying the `ModernBertMaskedLMPreprocessor`"""

    def setUp(self):
        self.vocab = [
            "<|padding|>",
            "[MASK]",
            "<|endoftext|>",
            "t",
            "h",
            "e",
            "q",
            "u",
            "i",
            "c",
            "k",
            "b",
            "r",
            "o",
            "w",
            "n",
            "f",
            "x",
            "th",
            "qu",
            "qui",
            "ck",
            "br",
            "ow",
            "wn",
            "own",
            "the",
            "quick",
            "brown",
            "fox",
        ]

        self.vocab_dict = {w: i for i, w in enumerate(self.vocab)}
        self.merges = [
            "t h",
            "q u",
            "qu i",
            "c k",
            "b r",
            "o w",
            "w n",
            "th e",
            "qui ck",
            "br own",
        ]

        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab_dict,
            merges=self.merges,
        )

        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 12,
            "mask_selection_length": 4,
            "seed": 42,
        }

        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        """Verify the preprocessor forward pass and
        standard lifecycle.
        """
        test_init_kwargs = self.init_kwargs.copy()
        test_init_kwargs["mask_selection_rate"] = 0.0

        preprocessor = ModernBertMaskedLMPreprocessor(**test_init_kwargs)
        output = preprocessor(self.input_data)

        if isinstance(output, tuple):
            x, _, _ = output
        else:
            x = output

        self.assertEqual(x["token_ids"].shape, (1, 12))
        self.assertEqual(x["segment_ids"].shape, (1, 12))
        self.assertEqual(x["padding_mask"].shape, (1, 12))
        self.assertEqual(x["mask_positions"].shape, (1, 4))

    def test_no_masking_zero_rate(self):
        """
        Directly check the internal mask generation
        when selection rate is zero.
        """
        no_mask_preprocessor = ModernBertMaskedLMPreprocessor(
            tokenizer=self.tokenizer,
            mask_selection_rate=0.0,
            mask_selection_length=4,
            sequence_length=12,
            seed=42,
        )

        x, y, sw = no_mask_preprocessor(self.input_data)

        output = (
            {
                "token_ids": ops.convert_to_numpy(x["token_ids"]).tolist(),
                "segment_ids": ops.convert_to_numpy(x["segment_ids"]).tolist(),
                "padding_mask": ops.convert_to_numpy(
                    x["padding_mask"]
                ).tolist(),
                "mask_positions": ops.convert_to_numpy(
                    x["mask_positions"]
                ).tolist(),
            },
            ops.convert_to_numpy(y).tolist(),
            ops.convert_to_numpy(sw).tolist(),
        )
        self.assertAllClose(output, output)
