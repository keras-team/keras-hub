"""Tests for BLIP-2 Flan-T5 SentencePiece tokenizer."""

import os

import pytest

from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_flan_t5_tokenizer import (
    BLIP2FlanT5Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class BLIP2FlanT5TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "proto": os.path.join(self.get_test_data_dir(), "t5_test_vocab.spm")
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=BLIP2FlanT5Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[4, 9, 5, 7], [4, 6, 8, 10]],
        )

    def test_backbone_cls(self):
        # The entire point of this class: backbone_cls must be BLIP2Backbone,
        # not T5Backbone, so from_preset() wires to the right backbone.
        self.assertEqual(BLIP2FlanT5Tokenizer.backbone_cls, BLIP2Backbone)

    def test_special_tokens(self):
        tokenizer = BLIP2FlanT5Tokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.end_token, "</s>")
        self.assertEqual(tokenizer.start_token, "</s>")
        self.assertEqual(tokenizer.pad_token, "<pad>")

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            BLIP2FlanT5Tokenizer(
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=BLIP2FlanT5Tokenizer,
            preset="blip2_flan_t5_xl",
            input_data=["Question: What is in this picture? Answer:"],
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BLIP2FlanT5Tokenizer.presets:
            self.run_preset_test(
                cls=BLIP2FlanT5Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
