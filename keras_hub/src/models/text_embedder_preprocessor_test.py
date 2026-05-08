import pytest

from keras_hub.src.models.bert.bert_text_embedder_preprocessor import (
    BertTextEmbedderPreprocessor,
)
from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
from keras_hub.src.models.text_embedder_preprocessor import (
    TextEmbedderPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class TestTextEmbedderPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTextEmbedderPreprocessor.presets.keys())
        gpt2_presets = set(GPT2Tokenizer.presets.keys())
        all_presets = set(TextEmbedderPreprocessor.presets.keys())
        self.assertTrue(bert_presets.issubset(all_presets))
        self.assertTrue(gpt2_presets.isdisjoint(all_presets))

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            TextEmbedderPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertTextEmbedderPreprocessor,
        )
        self.assertIsInstance(
            BertTextEmbedderPreprocessor.from_preset("bert_tiny_en_uncased"),
            BertTextEmbedderPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = TextEmbedderPreprocessor.from_preset(
            "bert_tiny_en_uncased", sequence_length=16
        )
        self.assertEqual(preprocessor.sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BertTextEmbedderPreprocessor.from_preset("gpt2_base_en")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            BertTextEmbedderPreprocessor.from_preset(
                "hf://spacy/en_core_web_sm"
            )
