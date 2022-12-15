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

from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_classifier import BertClassifier
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.bert.bert_tokenizer import BertTokenizer


@pytest.mark.large
class BertPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for BERT presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/bert/bert_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = BertTokenizer.from_preset(
            "bert_tiny_en_uncased",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [1996, 4248, 2829, 4419, 1012]
        self.assertAllEqual(outputs, expected_outputs)

    def test_preprocessor_output(self):
        tokenizer = BertPreprocessor.from_preset(
            "bert_tiny_en_uncased",
            sequence_length=4,
        )
        outputs = tokenizer("The quick brown fox.")["token_ids"]
        expected_outputs = [101, 1996, 4248, 102]
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
        model = BertBackbone.from_preset(
            "bert_tiny_en_uncased", load_weights=load_weights
        )
        outputs = model(input_data)["sequence_output"]
        if load_weights:
            # The forward pass from a preset should be stable!
            # This test should catch cases where we unintentionally change our
            # network code in a way that would invalidate our preset weights.
            # We should only update these numbers if we are updating a weights
            # file, or have found a discrepancy with the upstream source.
            outputs = outputs[0, 0, :5]
            expected = [-1.38173, 0.16598, -2.92788, -2.66958, -0.61556]
            # Keep a high tolerance, so we are robust to different hardware.
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_classifier_output(self, load_weights):
        input_data = tf.constant(["The quick brown fox."])
        model = BertClassifier.from_preset(
            "bert_tiny_en_uncased",
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
        model = BertClassifier.from_preset(
            "bert_tiny_en_uncased",
            load_weights=load_weights,
            preprocessor=None,
        )
        # Never assert output values, as the head weights are random.
        model.predict(input_data)

    @parameterized.named_parameters(
        ("bert_tokenizer", BertTokenizer),
        ("bert_preprocessor", BertPreprocessor),
        ("bert", BertBackbone),
        ("bert_classifier", BertClassifier),
    )
    def test_preset_mutability(self, cls):
        preset = "bert_tiny_en_uncased"
        obj = cls.from_preset(preset)
        # Cannot overwrite the presents attribute in an object
        with self.assertRaises(AttributeError):
            obj.presets = {"my_model": "clowntown"}
        # Cannot mutate presents in an object
        config = obj.presets[preset]["config"]
        config["num_layers"] = 1
        self.assertEqual(config["num_layers"], 1)
        self.assertEqual(obj.presets[preset]["config"]["num_layers"], 2)
        # Cannot mutate presets in the class
        config = BertBackbone.presets[preset]["config"]
        config["num_layers"] = 1
        self.assertEqual(config["num_layers"], 1)
        self.assertEqual(
            BertBackbone.presets[preset]["config"]["num_layers"], 2
        )

    @parameterized.named_parameters(
        ("bert_tokenizer", BertTokenizer),
        ("bert_preprocessor", BertPreprocessor),
        ("bert", BertBackbone),
        ("bert_classifier", BertClassifier),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("bert_tokenizer", BertTokenizer),
        ("bert_preprocessor", BertPreprocessor),
        ("bert", BertBackbone),
        ("bert_classifier", BertClassifier),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("bert_base_uncased_clowntown")

    def test_override_preprocessor_sequence_length(self):
        """Override sequence length longer than model's maximum."""
        preprocessor = BertPreprocessor.from_preset(
            "bert_base_en_uncased",
            sequence_length=64,
        )
        self.assertEqual(preprocessor.get_config()["sequence_length"], 64)
        preprocessor("The quick brown fox.")

    def test_override_preprocessor_sequence_length_gt_max(self):
        """Override sequence length longer than model's maximum."""
        with self.assertRaises(ValueError):
            BertPreprocessor.from_preset(
                "bert_base_en_uncased",
                sequence_length=1024,
            )


@pytest.mark.extra_large
class BertPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.

    This every presets for BERT and is only run manually.
    Run with:
    `pytest keras_nlp/models/bert_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_bert(self, load_weights):
        for preset in BertBackbone.presets:
            model = BertBackbone.from_preset(preset, load_weights=load_weights)
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
    def test_load_bert_classifier(self, load_weights):
        for preset in BertClassifier.presets:
            classifier = BertClassifier.from_preset(
                preset,
                load_weights=load_weights,
            )
            input_data = tf.constant(["This quick brown fox"])
            classifier.predict(input_data)

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_bert_classifier_without_preprocessing(self, load_weights):
        for preset in BertClassifier.presets:
            classifier = BertClassifier.from_preset(
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
                "segment_ids": tf.constant(
                    [0] * 200 + [1] * 312, shape=(1, 512)
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            classifier.predict(input_data)

    def test_load_tokenizers(self):
        for preset in BertTokenizer.presets:
            tokenizer = BertTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in BertPreprocessor.presets:
            preprocessor = BertPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
