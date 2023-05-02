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

from keras_nlp.models.albert.albert_backbone import AlbertBackbone
from keras_nlp.models.albert.albert_classifier import AlbertClassifier
from keras_nlp.models.albert.albert_preprocessor import AlbertPreprocessor
from keras_nlp.models.albert.albert_tokenizer import AlbertTokenizer


@pytest.mark.large
class AlbertPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for ALBERT presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/albert/albert_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = AlbertTokenizer.from_preset(
            "albert_base_en_uncased",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [13, 1, 438, 2231, 886, 2385, 9]
        self.assertAllEqual(outputs, expected_outputs)

    def test_preprocessor_output(self):
        preprocessor = AlbertPreprocessor.from_preset(
            "albert_base_en_uncased",
            sequence_length=4,
        )
        outputs = preprocessor("The quick brown fox.")["token_ids"]
        expected_outputs = [2, 13, 1, 3]
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_classifier_output(self, load_weights):
        input_data = tf.constant(["The quick brown fox."])
        model = AlbertClassifier.from_preset(
            "albert_base_en_uncased",
            num_classes=2,
            load_weights=load_weights,
        )
        # We don't assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_classifier_output_without_preprocessing(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "segment_ids": tf.constant([[0, 0, 0, 0]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = AlbertClassifier.from_preset(
            "albert_base_en_uncased",
            num_classes=2,
            load_weights=load_weights,
            preprocessor=None,
        )
        # Never assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[2, 13, 1, 3]]),
            "segment_ids": tf.constant([[0, 0, 0, 0]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = AlbertBackbone.from_preset(
            "albert_base_en_uncased", load_weights=load_weights
        )
        outputs = model(input_data)
        if load_weights:
            outputs = outputs["sequence_output"][0, 0, :5]
            expected = [1.830863, 1.698645, -1.819195, -0.53382, -0.38114]
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("albert_tokenizer", AlbertTokenizer),
        ("albert_preprocessor", AlbertPreprocessor),
        ("albert", AlbertBackbone),
        ("albert_classifier", AlbertClassifier),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("albert_tokenizer", AlbertTokenizer, {}),
        ("albert_preprocessor", AlbertPreprocessor, {}),
        ("albert", AlbertBackbone, {}),
        ("albert_classifier", AlbertClassifier, {"num_classes": 2}),
    )
    def test_unknown_preset_error(self, cls, kwargs):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("albert_base_en_uncased_clowntown", **kwargs)


@pytest.mark.extra_large
class AlbertPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This tests every ALBERT preset and is only run manually.
    Run with:
    `pytest keras_nlp/models/albert/albert_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_albert(self, load_weights):
        for preset in AlbertBackbone.presets:
            model = AlbertBackbone.from_preset(
                preset, load_weights=load_weights
            )
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
                ),
                "segment_ids": tf.constant(
                    [0] * 200 + [1] * 312, shape=(1, 512)
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            model(input_data)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_albert_classifier(self, load_weights):
        for preset in AlbertClassifier.presets:
            classifier = AlbertClassifier.from_preset(
                preset,
                num_classes=2,
                load_weights=load_weights,
            )
            input_data = tf.constant(["This quick brown fox"])
            classifier.predict(input_data)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_albert_classifier_without_preprocessing(self, load_weights):
        for preset in AlbertClassifier.presets:
            classifier = AlbertClassifier.from_preset(
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
        for preset in AlbertTokenizer.presets:
            tokenizer = AlbertTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in AlbertPreprocessor.presets:
            preprocessor = AlbertPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
