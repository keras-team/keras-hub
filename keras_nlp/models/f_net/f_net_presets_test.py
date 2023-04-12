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
from keras_nlp.models.f_net.f_net_classifier import FNetClassifier
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
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "segment_ids": tf.constant([[0, 0, 0, 0]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = FNetBackbone.from_preset(
            "f_net_base_en", load_weights=load_weights
        )
        outputs = model(input_data)["sequence_output"]
        if load_weights:
            # The forward pass from a preset should be stable!
            # This test should catch cases where we unintentionally change our
            # network code in a way that would invalidate our preset weights.
            # We should only update these numbers if we are updating a weights
            # file, or have found a discrepancy with the upstream source.
            outputs = outputs[0, 0, :5]
            expected = [4.157282, -0.096616, -0.244943, -0.068104, -0.559592]
            # Keep a high tolerance, so we are robust to different hardware.
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_classifier_output(self, load_weights):
        input_data = tf.constant(["The quick brown fox."])
        model = FNetClassifier.from_preset(
            "f_net_base_en",
            num_classes=2,
            load_weights=load_weights,
        )
        # We don't assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("f_net_tokenizer", FNetTokenizer),
        ("f_net_preprocessor", FNetPreprocessor),
        ("f_net", FNetBackbone),
        ("f_net_classifier", FNetClassifier),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("f_net_tokenizer", FNetTokenizer, {}),
        ("f_net_preprocessor", FNetPreprocessor, {}),
        ("f_net", FNetBackbone, {}),
        ("f_net_classifier", FNetClassifier, {"num_classes": 2}),
    )
    def test_unknown_preset_error(self, cls, kwargs):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("f_net_base_en_clowntown", **kwargs)


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

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_fnet_classifier(self, load_weights):
        for preset in FNetClassifier.presets:
            classifier = FNetClassifier.from_preset(
                preset,
                num_classes=2,
                load_weights=load_weights,
            )
            input_data = tf.constant(["This quick brown fox"])
            classifier.predict(input_data)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_fnet_classifier_without_preprocessing(self, load_weights):
        for preset in FNetClassifier.presets:
            classifier = FNetClassifier.from_preset(
                preset,
                num_classes=2,
                preprocessor=None,
                load_weights=load_weights,
            )
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512),
                    dtype=tf.int64,
                    maxval=classifier.backbone.vocabulary_size,
                ),
                "segment_ids": tf.constant(
                    [0] * 200 + [1] * 312, shape=(1, 512)
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            classifier.predict(input_data)

    def test_load_tokenizers(self):
        for preset in FNetTokenizer.presets:
            tokenizer = FNetTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in FNetPreprocessor.presets:
            preprocessor = FNetPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
