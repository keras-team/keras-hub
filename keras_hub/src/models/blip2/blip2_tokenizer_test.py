"""Tests for BLIP-2 tokenizer."""

import pytest

from keras_hub.src.models.blip2.blip2_tokenizer import Blip2Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Blip2TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary": {
                "<pad>": 1,
                "</s>": 2,
                "<image>": 3,
                "\u010a": 4,
                "Ġ": 5,
                "t": 6,
                "h": 7,
                "e": 8,
                "q": 9,
                "u": 10,
                "i": 11,
                "c": 12,
                "k": 13,
                "th": 14,
            },
            "merges": ["t h"],
        }
        self.input_data = ["the", "the quick"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Blip2Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[14, 8], [14, 8, 5, 9, 10, 11, 12, 13]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Blip2Tokenizer(
                vocabulary={"<pad>": 1, "</s>": 2, "a": 3},
                merges=["a b"],
            )

        with self.assertRaises(ValueError):
            Blip2Tokenizer(
                vocabulary={"<pad>": 1, "</s>": 2, "\u010a": 3, "a": 4},
                merges=["a b"],
            )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=Blip2Tokenizer,
            preset="blip2_opt_2_7b",
            input_data=["Question: What is this? Answer:"],
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Blip2Tokenizer.presets:
            self.run_preset_test(
                cls=Blip2Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
