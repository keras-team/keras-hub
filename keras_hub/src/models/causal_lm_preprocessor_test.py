import pytest

from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_hub.src.tests.test_case import TestCase
from keras import ops


class TestCausalLMPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTokenizer.presets.keys())
        gpt2_presets = set(GPT2Preprocessor.presets.keys())
        all_presets = set(CausalLMPreprocessor.presets.keys())
        self.assertTrue(bert_presets.isdisjoint(all_presets))
        self.assertTrue(gpt2_presets.issubset(all_presets))

    def test_padding_side_call(self):
        preprocessor = CausalLMPreprocessor.from_preset(
            "gpt2_base_en", sequence_length=7
        )
        # left pad
        outputs = preprocessor(
            ["i love you", "this is keras hub"], padding_side="left"
        )

        self.assertAllEqual(
            outputs[0]["token_ids"],
            (
                [0, 0, 50256, 72, 1842, 345, 50256],
                [50256, 5661, 318, 41927, 292, 12575, 50256],
            ),
        )
        self.assertAllEqual(
            outputs[1],
            (
                [0, 50256, 72, 1842, 345, 50256, 0],
                [5661, 318, 41927, 292, 12575, 50256, 0],
            ),
        )
        self.assertAllEqual(
            outputs[2],
            (
                [False, True, True, True, True, True, False],
                [True, True, True, True, True, True, False],
            ),
        )
        # right pad
        outputs = preprocessor(
            ["i love you", "this is keras hub"], padding_side="right"
        )
        self.assertAllEqual(
            outputs[0]["token_ids"],
            (
                [50256, 72, 1842, 345, 50256, 0, 0],
                [50256, 5661, 318, 41927, 292, 12575, 50256],
            ),
        )
        self.assertAllEqual(
            outputs[1],
            (
                [72, 1842, 345, 50256, 0, 0, 0],
                [5661, 318, 41927, 292, 12575, 50256, 0],
            ),
        )
        self.assertAllEqual(
            outputs[2],
            (
                [ True,  True,  True,  True, False, False, False],
                [True, True, True, True, True, True, False],
            ),
        )

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            CausalLMPreprocessor.from_preset("gpt2_base_en"),
            GPT2CausalLMPreprocessor,
        )
        self.assertIsInstance(
            GPT2CausalLMPreprocessor.from_preset("gpt2_base_en"),
            GPT2CausalLMPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = CausalLMPreprocessor.from_preset(
            "gpt2_base_en", sequence_length=16
        )
        self.assertEqual(preprocessor.sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            GPT2CausalLMPreprocessor.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            GPT2CausalLMPreprocessor.from_preset("hf://spacy/en_core_web_sm")
