import os

import pytest

from keras_hub.src.models.modernbert.modernbert_preprocessor import (
    ModernBertMaskedLMPreprocessor,
)
from keras_hub.src.models.modernbert.modernbert_tokenizer import (
    ModernBertTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertMaskedLMPreprocessorTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "hidden_dim": 8,
            "intermediate_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "local_attention_window": 128,
            "global_attn_every_n_layers": 2,
            "dropout": 0.0,
        }
        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=ModernBertMaskedLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                    "mask_positions": [[1, 2, 3, 4]],
                },
                [[5, 10, 6, 8]],
                [[1.0, 1.0, 1.0, 1.0]],
            ),
        )

    def test_no_masking_zero_rate(self):
        no_mask_preprocessor = ModernBertMaskedLMPreprocessor(
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
                    "token_ids": [[1, 5, 10, 6, 8, 2, 0, 0, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],
                    "mask_positions": [[0, 0, 0, 0]],
                },
                [[0, 0, 0, 0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ),
        )
