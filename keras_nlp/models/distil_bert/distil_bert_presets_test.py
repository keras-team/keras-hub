# Copyright 2022 The KerasNLP Authors
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

from keras_nlp.models.distil_bert.distil_bert_backbone import DistilBertBackbone
from keras_nlp.models.distil_bert.distil_bert_classifier import (
    DistilBertClassifier,
)
from keras_nlp.models.distil_bert.distil_bert_preprocessor import (
    DistilBertPreprocessor,
)
from keras_nlp.models.distil_bert.distil_bert_tokenizer import (
    DistilBertTokenizer,
)


@pytest.mark.large
class DistilBertPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for DistilBERT presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/distilbert/distilbert_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = DistilBertTokenizer.from_preset(
            "distil_bert_base_en_uncased",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [1996, 4248, 2829, 4419, 1012]
        self.assertAllEqual(outputs, expected_outputs)

    def test_preprocessor_output(self):
        tokenizer = DistilBertPreprocessor.from_preset(
            "distil_bert_base_en_uncased",
            sequence_length=4,
        )
        outputs = tokenizer("The quick brown fox.")["token_ids"]
        expected_outputs = [101, 1996, 4248, 102]
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = DistilBertBackbone.from_preset(
            "distil_bert_base_en_uncased", load_weights=load_weights
        )
        outputs = model(input_data)[0, 0, :5]
        if load_weights:
            expected_outputs = [-0.2381, -0.1965, 0.1053, -0.0847, -0.145]
            self.assertAllClose(outputs, expected_outputs, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_classifier_output(self, load_weights):
        input_data = tf.constant(["The quick brown fox."])
        model = DistilBertClassifier.from_preset(
            "distil_bert_base_en_uncased",
            load_weights=load_weights,
        )
        model.predict(input_data)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_classifier_output_without_preprocessing(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = DistilBertClassifier.from_preset(
            "distil_bert_base_en_uncased",
            load_weights=load_weights,
            preprocessor=None,
        )
        model.predict(input_data)

    @parameterized.named_parameters(
        ("distilbert_tokenizer", DistilBertTokenizer),
        ("distilbert_preprocessor", DistilBertPreprocessor),
        ("distilbert", DistilBertBackbone),
        ("distilbert_classifier", DistilBertClassifier),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("distilbert_tokenizer", DistilBertTokenizer),
        ("distilbert_preprocessor", DistilBertPreprocessor),
        ("distilbert", DistilBertBackbone),
        ("distilbert_classifier", DistilBertClassifier),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("distilbert_base_uncased_clowntown")


@pytest.mark.extra_large
class DistilBertPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Tests the full enumeration of our preset.

    This tests every DistilBERT preset and is only run manually.
    Run with:
    `pytest keras_nlp/models/distilbert/distilbert_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_distilbert(self, load_weights):
        for preset in DistilBertBackbone.presets:
            model = DistilBertBackbone.from_preset(
                preset, load_weights=load_weights
            )
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            model(input_data)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_distilbert_classifier(self, load_weights):
        for preset in DistilBertClassifier.presets:
            classifier = DistilBertClassifier.from_preset(
                preset,
                num_classes=4,
                load_weights=load_weights,
            )
            input_data = tf.constant(["This quick brown fox"])
            classifier.predict(input_data)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_distilbert_classifier_no_preprocessing(self, load_weights):
        for preset in DistilBertClassifier.presets:
            classifier = DistilBertClassifier.from_preset(
                preset,
                num_classes=4,
                load_weights=load_weights,
                preprocessor=None,
            )
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512),
                    dtype=tf.int64,
                    maxval=classifier.backbone.vocabulary_size,
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            classifier.predict(input_data)

    def test_load_tokenizers(self):
        for preset in DistilBertTokenizer.presets:
            tokenizer = DistilBertTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in DistilBertPreprocessor.presets:
            preprocessor = DistilBertPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
