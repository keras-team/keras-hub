import pytest

from keras_hub.src.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.tests.test_case import TestCase


class TestSeq2SeqLMPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTokenizer.presets.keys())
        bart_presets = set(BartSeq2SeqLMPreprocessor.presets.keys())
        all_presets = set(Seq2SeqLMPreprocessor.presets.keys())
        self.assertTrue(bert_presets.isdisjoint(all_presets))
        self.assertTrue(bart_presets.issubset(all_presets))

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            Seq2SeqLMPreprocessor.from_preset("bart_base_en"),
            BartSeq2SeqLMPreprocessor,
        )
        self.assertIsInstance(
            BartSeq2SeqLMPreprocessor.from_preset("bart_base_en"),
            BartSeq2SeqLMPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = Seq2SeqLMPreprocessor.from_preset(
            "bart_base_en", decoder_sequence_length=16
        )
        self.assertEqual(preprocessor.decoder_sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            BartSeq2SeqLMPreprocessor.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            BartSeq2SeqLMPreprocessor.from_preset("hf://spacy/en_core_web_sm")
