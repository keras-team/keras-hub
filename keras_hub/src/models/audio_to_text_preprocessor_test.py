import pytest

from keras_hub.src.models.audio_to_text_preprocessor import (
    AudioToTextPreprocessor,
)
from keras_hub.src.models.bert.bert_tokenizer import BertTokenizer
from keras_hub.src.models.moonshine.moonshine_audio_to_text_preprocessor import (  # noqa: E501
    MoonshineAudioToTextPreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class TestAudioToTextPreprocessor(TestCase):
    def test_preset_accessors(self):
        bert_presets = set(BertTokenizer.presets.keys())
        moonshine_presets = set(MoonshineAudioToTextPreprocessor.presets.keys())
        all_presets = set(AudioToTextPreprocessor.presets.keys())
        self.assertTrue(bert_presets.isdisjoint(all_presets))
        self.assertTrue(moonshine_presets.issubset(all_presets))

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            AudioToTextPreprocessor.from_preset("moonshine_base_en"),
            MoonshineAudioToTextPreprocessor,
        )
        self.assertIsInstance(
            MoonshineAudioToTextPreprocessor.from_preset("moonshine_base_en"),
            MoonshineAudioToTextPreprocessor,
        )

    @pytest.mark.large
    def test_from_preset_with_sequence_length(self):
        preprocessor = AudioToTextPreprocessor.from_preset(
            "moonshine_base_en", decoder_sequence_length=16
        )
        self.assertEqual(preprocessor.decoder_sequence_length, 16)

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            # No loading on an incorrect class.
            MoonshineAudioToTextPreprocessor.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            MoonshineAudioToTextPreprocessor.from_preset(
                "hf://spacy/en_core_web_sm"
            )
