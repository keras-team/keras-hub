"""Tests for MetaCLIP 2 tokenizer."""

import os

import pytest

from keras_hub.src.models.metaclip_2.metaclip_2_tokenizer import (
    MetaCLIP2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class MetaCLIP2TokenizerTest(TestCase):
    def setUp(self):
        # Use the XLM-RoBERTa test vocab since MetaCLIP 2 uses XLM-V
        # which is based on XLM-RoBERTa architecture
        self.init_kwargs = {
            "proto": os.path.join(
                self.get_test_data_dir(), "xlm_roberta_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MetaCLIP2Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[6, 11, 7, 9], [6, 8, 10, 12]],
        )

    def test_special_tokens(self):
        tokenizer = MetaCLIP2Tokenizer(**self.init_kwargs)
        # XLM-V/XLM-RoBERTa style special tokens
        self.assertEqual(tokenizer.start_token, "<s>")
        self.assertEqual(tokenizer.end_token, "</s>")
        self.assertEqual(tokenizer.pad_token, "<pad>")
        self.assertEqual(tokenizer.unk_token, "<unk>")
        self.assertEqual(tokenizer.start_token_id, 0)
        self.assertEqual(tokenizer.end_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertEqual(tokenizer.unk_token_id, 3)

    def test_special_token_properties(self):
        tokenizer = MetaCLIP2Tokenizer(**self.init_kwargs)
        self.assertEqual(
            tokenizer.special_tokens, ["<s>", "<pad>", "</s>", "<unk>"]
        )
        self.assertEqual(tokenizer.special_token_ids, [0, 1, 2, 3])

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MetaCLIP2Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
