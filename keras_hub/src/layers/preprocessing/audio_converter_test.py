# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib

import numpy as np
import pytest

from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.whisper.whisper_audio_converter import (
    WhisperAudioConverter,
)
from keras_hub.src.tests.test_case import TestCase


class AudioConverterTest(TestCase):
    def test_preset_accessors(self):
        whisper_presets = set(WhisperAudioConverter.presets.keys())
        all_presets = set(AudioConverter.presets.keys())
        self.assertContainsSubset(whisper_presets, all_presets)
        self.assertIn("whisper_tiny_en", whisper_presets)
        self.assertIn("whisper_tiny_en", all_presets)

    @pytest.mark.large
    def test_from_preset(self):
        self.assertIsInstance(
            AudioConverter.from_preset("whisper_tiny_en"),
            WhisperAudioConverter,
        )

    @pytest.mark.large
    def test_from_preset_errors(self):
        with self.assertRaises(ValueError):
            AudioConverter.from_preset("bert_tiny_en_uncased")
        with self.assertRaises(ValueError):
            # No loading on a non-keras model.
            AudioConverter.from_preset("hf://spacy/en_core_web_sm")

    @pytest.mark.large
    def test_save_to_preset(self):
        save_dir = self.get_temp_dir()
        converter = AudioConverter.from_preset(
            "whisper_tiny_en",
            num_mels=40,
        )
        converter.save_to_preset(save_dir)
        # Save a backbone so the preset is valid.
        backbone = Backbone.from_preset("whisper_tiny_en", load_weights=False)
        backbone.save_to_preset(save_dir)

        # Check existence of files.
        path = pathlib.Path(save_dir)
        self.assertTrue(os.path.exists(path / "audio_converter.json"))

        # Check loading.
        restored = AudioConverter.from_preset(save_dir)
        test_audio = np.random.rand(1_000)
        self.assertAllClose(restored(test_audio), converter(test_audio))
