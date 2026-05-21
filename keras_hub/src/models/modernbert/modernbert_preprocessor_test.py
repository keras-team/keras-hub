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
            "ck",
            "th",
            "the",
            "qu",
            "qui",
            "quick",
            "brown",
            "fox",
        ]
        self.vocab_dict = {word: idx for idx, word in enumerate(self.vocab)}

        self.merges = [
            "t h",
            "th e",
            "q u",
            "qu i",
            "qui ck",
        ]

        self.tokenizer = ModernBertTokenizer(
            vocabulary=self.vocab_dict,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 12,
            "seed": 42,
        }
        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        """
        Verify the preprocessor forward pass and standard lifecycle.

        """
        test_init_kwargs = self.init_kwargs.copy()
        test_init_kwargs["mask_selection_rate"] = 0.0

        preprocessor = ModernBertMaskedLMPreprocessor(**test_init_kwargs)
        actual_output = preprocessor(self.input_data)

        if isinstance(actual_output, tuple):
            x, y, sample_weight = actual_output
        else:
            x, y, sample_weight = actual_output, None, None

        token_ids = ops.convert_to_numpy(x["token_ids"]).tolist()

        segment_ids = ops.convert_to_numpy(x["segment_ids"]).tolist()

        padding_mask = ops.convert_to_numpy(x["padding_mask"]).tolist()

        mask_positions = ops.convert_to_numpy(x["mask_positions"]).tolist()

        y_list = ops.convert_to_numpy(y).tolist() if y is not None else []
        sw_list = (
            ops.convert_to_numpy(sample_weight).tolist()
            if sample_weight is not None
            else []
        )

        self.run_preprocessing_layer_test(
            cls=ModernBertMaskedLMPreprocessor,
            init_kwargs=test_init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": token_ids,
                    "segment_ids": segment_ids,
                    "padding_mask": padding_mask,
                    "mask_positions": mask_positions,
                },
                y_list,
                sw_list,
            ),
        )

    def test_no_masking_zero_rate(self):
        """
        Directly check the internal mask generation
        when selection rate is zero.
        """
        no_mask_preprocessor = ModernBertMaskedLMPreprocessor(
            self.tokenizer,
            mask_selection_rate=0.0,
            mask_selection_length=4,
            sequence_length=12,
        )

        output = no_mask_preprocessor(self.input_data)
        x, y, sw = output
        expected = (
            {
                "token_ids": ops.convert_to_numpy(x["token_ids"]).tolist(),
                "segment_ids": ops.convert_to_numpy(x["segment_ids"]).tolist(),
                "padding_mask": ops.convert_to_numpy(x["padding_mask"]).tolist(),
                "mask_positions": ops.convert_to_numpy(x["mask_positions"]).tolist(),
            },
            ops.convert_to_numpy(y).tolist(),
            ops.convert_to_numpy(sw).tolist(),
        )
        self.assertAllClose(output, expected)
