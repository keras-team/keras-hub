import pytest

from keras_hub.src.models.modernbert.modern_bert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertTokenizerTest(TestCase):
    """
    Tests for verifying the `ModernBertTokenizer`
    implementation details.
    """

    def setUp(self):
        self.merges = [
            "Ġ a",
            "Ġ t",
            "Ġ i",
            "Ġ b",
            "a i",
            "p l",
            "n e",
        ]
        self.merges += [
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
        ]
        self.merges += [
            "Ġai r",
            "Ġa i",
            "pla ne",
        ]

        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])

        self.vocab = sorted(set(self.vocab))
        self.vocab += [
            "<|endoftext|>",
            "<|padding|>",
            "[MASK]",
            "[UNK]",
        ]
        self.vocab = {token: i for i, token in enumerate(self.vocab)}

        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
        }

        self.input_data = [
            "<|endoftext|> airplane at airport",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=ModernBertTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [29, 23, 14, 24, 23, 16],
                [23, 14, 23, 16],
            ],
            expected_detokenize_output=[
                "<|endoftext|> airplane at airport",
                " airplane airport",
            ],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            ModernBertTokenizer(
                vocabulary=["a", "b", "c"],
                merges=[],
            )

    def test_special_token_ids(self):
        tokenizer = ModernBertTokenizer(**self.init_kwargs)

        self.assertEqual(
            tokenizer.start_token_id,
            tokenizer.cls_token_id,
        )
        self.assertEqual(
            tokenizer.end_token_id,
            tokenizer.sep_token_id,
        )
        self.assertEqual(
            tokenizer.start_token_id,
            tokenizer.end_token_id,
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=ModernBertTokenizer,
            preset="modernbert_base_en",
            input_data=["The quick brown fox."],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ModernBertTokenizer.presets:
            self.run_preset_test(
                cls=ModernBertTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
