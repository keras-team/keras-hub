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

from keras_nlp.models.roberta.roberta_backbone import RobertaBackbone
from keras_nlp.models.roberta.roberta_classifier import RobertaClassifier
from keras_nlp.models.roberta.roberta_preprocessor import RobertaPreprocessor
from keras_nlp.models.roberta.roberta_tokenizer import RobertaTokenizer


@pytest.mark.large
class RobertaPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for RoBERTa presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/roberta/roberta_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = RobertaTokenizer.from_preset(
            "roberta_base",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [133, 2119, 6219, 23602, 4]
        self.assertAllEqual(outputs, expected_outputs)

    def test_preprocessor_output(self):
        preprocessor = RobertaPreprocessor.from_preset(
            "roberta_base",
            sequence_length=4,
        )
        outputs = preprocessor("The quick brown fox.")["token_ids"]
        expected_outputs = [0, 133, 2119, 2]
        self.assertAllEqual(outputs, expected_outputs)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[0, 133, 2119, 2]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = RobertaBackbone.from_preset(
            "roberta_base", load_weights=load_weights
        )
        outputs = model(input_data)
        if load_weights:
            outputs = outputs[0, 0, :5]
            expected = [-0.051, 0.100, -0.010, -0.097, 0.059]
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_classifier_output(self, load_weights):
        input_data = ["Let's rock!"]
        model = RobertaClassifier.from_preset(
            "roberta_base", load_weights=load_weights
        )
        # Never assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_classifier_output_without_preprocessing(self, load_weights):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = RobertaClassifier.from_preset(
            "roberta_base",
            load_weights=load_weights,
            preprocessor=None,
        )
        # Never assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("roberta_tokenizer", RobertaTokenizer),
        ("roberta_preprocessor", RobertaPreprocessor),
        ("roberta", RobertaBackbone),
        ("roberta_classifier", RobertaClassifier),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("roberta_tokenizer", RobertaTokenizer),
        ("roberta_preprocessor", RobertaPreprocessor),
        ("roberta", RobertaBackbone),
        ("roberta_classifier", RobertaClassifier),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("roberta_base_clowntown")


@pytest.mark.extra_large
class RobertaPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.

    This tests every RoBERTa preset and is only run manually.
    Run with:
    `pytest keras_nlp/models/roberta/roberta_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("preset_weights", True), ("random_weights", False)
    )
    def test_load_roberta(self, load_weights):
        for preset in RobertaBackbone.presets:
            model = RobertaBackbone.from_preset(
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
    def test_load_roberta_classifier(self, load_weights):
        for preset in RobertaClassifier.presets:
            classifier = RobertaClassifier.from_preset(
                preset, num_classes=4, load_weights=load_weights
            )
            input_data = ["The quick brown fox."]
            classifier(input_data)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_roberta_classifier_without_preprocessing(self, load_weights):
        for preset in RobertaClassifier.presets:
            classifier = RobertaClassifier.from_preset(
                preset,
                preprocessor=None,
                load_weights=load_weights,
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
        for preset in RobertaTokenizer.presets:
            tokenizer = RobertaTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in RobertaPreprocessor.presets:
            preprocessor = RobertaPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
