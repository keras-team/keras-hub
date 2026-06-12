import keras
import pytest

from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.whisper.whisper_audio_to_text import (
    WhisperAudioToText,
)
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = WhisperAudioToText.from_preset("hf://openai/whisper-tiny.en")
        audio = keras.random.normal((1, 16000))
        model.generate({"audio": audio}, max_length=10)

    @pytest.mark.large
    def test_class_detection(self):
        model = AudioToText.from_preset(
            "hf://openai/whisper-tiny.en",
            load_weights=False,
        )
        self.assertIsInstance(model, WhisperAudioToText)
        model = Backbone.from_preset(
            "hf://openai/whisper-tiny.en",
            load_weights=False,
        )
        self.assertIsInstance(model, WhisperBackbone)

    # TODO: compare numerics with huggingface model
