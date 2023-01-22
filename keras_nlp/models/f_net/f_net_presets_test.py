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
"""Tests for loading pretrained model presets."""

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.models.f_net.f_net_backbone import FNetBackbone
from keras_nlp.models.f_net.f_net_preprocessor import FNetPreprocessor
from keras_nlp.models.f_net.f_net_tokenizer import FNetTokenizer


@pytest.mark.large
class FNetPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for FNet presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/f_net/f_net_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = FNetTokenizer.from_preset(
            "f_net_base_en",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [97, 1467, 5187, 26, 2521, 16678]
        self.assertAllEqual(outputs, expected_outputs)

    def test_preprocessor_output(self):
        preprocessor = FNetPreprocessor.from_preset(
            "f_net_base_en",
            sequence_length=4,
        )
        outputs = preprocessor("The quick brown fox.")["token_ids"]
        expected_outputs = [4, 97, 1467, 5]
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[4, 97, 1467, 5]]),
            "segment_ids": tf.constant([[0, 0, 0, 0]]),
        }
        model = FNetBackbone.from_preset(
            "f_net_base_en", load_weights=load_weights
        )
        outputs = model(input_data)
        if load_weights:
            outputs = outputs["sequence_output"][0, 0, :5]
            expected = [4.182479, -0.072181, -0.138097, -0.036582, -0.521765]
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("f_net_tokenizer", FNetTokenizer),
        ("f_net_preprocessor", FNetPreprocessor),
        ("f_net", FNetBackbone),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("f_net_tokenizer", FNetTokenizer),
        ("f_net_preprocessor", FNetPreprocessor),
        ("f_net", FNetBackbone),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("f_net_base_en_clowntown")


@pytest.mark.extra_large
class FNetPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.

    This tests every FNet preset and is only run manually.
    Run with:
    `pytest keras_nlp/models/f_net/f_net_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_f_net(self, load_weights):
        for preset in FNetBackbone.presets:
            model = FNetBackbone.from_preset(preset, load_weights=load_weights)
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
                ),
                "segment_ids": tf.constant(
                    [0] * 200 + [1] * 312, shape=(1, 512)
                ),
            }
            model(input_data)

    def test_load_tokenizers(self):
        for preset in FNetTokenizer.presets:
            tokenizer = FNetTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in FNetPreprocessor.presets:
            preprocessor = FNetPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
