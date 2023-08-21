# Copyright 2023 The KerasNLP Authors
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

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.models.whisper.whisper_audio_feature_extractor import (
    WhisperAudioFeatureExtractor,
)
from keras_nlp.models.whisper.whisper_backbone import WhisperBackbone
from keras_nlp.models.whisper.whisper_tokenizer import WhisperTokenizer


@pytest.mark.tf_only
@pytest.mark.large
class WhisperPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for Whisper presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/whisper/whisper_presets_test.py --run_large`
    """

    def test_audio_feature_extractor_output(self):
        audio_feature_extractor = WhisperAudioFeatureExtractor.from_preset(
            "whisper_tiny_en"
        )
        # Don't really need to check for output here.
        audio_feature_extractor(tf.ones((200,)))

    def test_tokenizer_output(self):
        tokenizer = WhisperTokenizer.from_preset("whisper_tiny_en")
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [464, 2068, 7586, 21831, 13]
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    @pytest.mark.skip  # TODO: fix weight mismatch error.
    def test_backbone_output(self, load_weights):
        input_data = {
            "encoder_features": tf.ones((1, 3000, 80)),
            "decoder_token_ids": tf.constant(
                [[50257, 50362, 464, 2068, 7586, 21831, 13, 50256, 50256]]
            ),
            "decoder_padding_mask": tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
        }
        model = WhisperBackbone.from_preset(
            "whisper_tiny_en", load_weights=load_weights
        )
        outputs = model(input_data)["decoder_sequence_output"][0, 0, :5]
        if load_weights:
            # The forward pass from a preset should be stable!
            # This test should catch cases where we unintentionally change our
            # network code in a way that would invalidate our preset weights.
            # We should only update these numbers if we are updating a weights
            # file, or have found a discrepancy with the upstream source.
            expected_outputs = [13.238, 1.051, 8.348, -20.012, -5.022]
            # Keep a high tolerance, so we are robust to different hardware.
            self.assertAllClose(outputs, expected_outputs, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("whisper_tokenizer", WhisperTokenizer),
        ("whisper", WhisperBackbone),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("whisper_tokenizer", WhisperTokenizer),
        ("whisper", WhisperBackbone),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("whisper_tiny_en_clowntown")


@pytest.mark.extra_large
class WhisperPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.

    This tests every Whisper preset and is only run manually.
    Run with:
    `pytest keras_nlp/models/whisper/whisper_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_whisper(self, load_weights):
        for preset in WhisperBackbone.presets:
            model = WhisperBackbone.from_preset(
                preset, load_weights=load_weights
            )
            input_data = {
                "encoder_features": tf.ones((1, 3000, 80)),
                "decoder_token_ids": tf.random.uniform(
                    shape=(1, 446),
                    dtype="int64",
                    maxval=model.vocabulary_size,
                ),
                "decoder_padding_mask": tf.constant([1] * 446, shape=(1, 446)),
            }
            model(input_data)

    def test_load_tokenizers(self):
        for preset in WhisperTokenizer.presets:
            tokenizer = WhisperTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_audio_feature_extractors(self):
        for preset in WhisperAudioFeatureExtractor.presets:
            audio_feature_extractor = WhisperAudioFeatureExtractor.from_preset(
                preset
            )
            audio_feature_extractor(tf.ones((200,)))
