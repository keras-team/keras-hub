import os

import pytest

from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_preprocessor import GPT2Preprocessor
from keras_hub.src.tests.test_case import TestCase


class TestCausalLMPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTokenizer.presets.keys())
        gpt2_presets = set(GPT2Preprocessor.presets.keys())
        all_presets = set(CausalLMPreprocessor.presets.keys())
        self.assertTrue(bert_presets.isdisjoint(all_presets))
        self.assertTrue(gpt2_presets.issubset(all_presets))

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

    def test_export_supported_preprocessor(self):
        proto = os.path.join(
            os.path.dirname(__file__),
            "../tests/test_data",
            "gemma_export_vocab.spm",
        )
        tokenizer = GemmaTokenizer(proto=proto)
        preprocessor = GemmaCausalLMPreprocessor(tokenizer=tokenizer)
        export_path = os.path.join(self.get_temp_dir(), "export_preprocessor")
        preprocessor.export_to_transformers(export_path)
        # Basic check: tokenizer config exists
        self.assertTrue(
            os.path.exists(os.path.join(export_path, "tokenizer_config.json"))
        )

    def test_export_missing_tokenizer(self):
        preprocessor = GemmaCausalLMPreprocessor(tokenizer=None)
        export_path = os.path.join(
            self.get_temp_dir(), "export_missing_tokenizer"
        )
        with self.assertRaises(ValueError):
            preprocessor.export_to_transformers(export_path)
