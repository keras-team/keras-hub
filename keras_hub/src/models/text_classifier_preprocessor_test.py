import pytest

from keras_hub.src.models.bert.bert_text_classifier_preprocessor import (
    BertTextClassifierPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_hub.src.models.text_classifier_preprocessor import (
    TextClassifierPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class TestTextClassifierPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTextClassifierPreprocessor.presets.keys())
        gpt2_presets = set(GPT2Tokenizer.presets.keys())
        all_presets = set(TextClassifierPreprocessor.presets.keys())
        self.assertTrue(bert_presets.issubset(all_presets))
        self.assertTrue(gpt2_presets.isdisjoint(all_presets))

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            TextClassifierPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertTextClassifierPreprocessor,
        )
        self.assertIsInstance(
            BertTextClassifierPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertTextClassifierPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = TextClassifierPreprocessor.from_preset(
            "bert_tiny_en_uncased", sequence_length=16
        )
        self.assertEqual(preprocessor.sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BertTextClassifierPreprocessor.from_preset("gpt2_base_en")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            BertTextClassifierPreprocessor.from_preset(
                "hf://spacy/en_core_web_sm"
            )
