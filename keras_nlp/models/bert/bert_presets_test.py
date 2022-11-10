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

from keras_nlp.models.bert.bert_models import Bert
from keras_nlp.models.bert.bert_preprocessing import BertPreprocessor
from keras_nlp.models.bert.bert_tasks import BertClassifier


@pytest.mark.large
class BertPresetSmokeTest(tf.test.TestCase):
    """
    A smoke test for BERT presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/bert/bert_presets_test.py --run_large`
    """

    def test_preprocessor_output(self):
        tokenizer = BertPreprocessor.from_preset(
            "bert_tiny_uncased_en",
            sequence_length=4,
        )
        outputs = tokenizer("The quick brown fox.")["token_ids"]
        expected_outputs = [101, 1996, 4248, 102]
        self.assertAllEqual(outputs, expected_outputs)

    def test_backbone_output(self):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "segment_ids": tf.constant([[0, 0, 0, 0]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = Bert.from_preset(
            "bert_tiny_uncased_en",
        )
        outputs = model(input_data)["sequence_output"][0, 0, :5]
        # Outputs from our preset checkpoints should be stable!
        # We should only update these numbers if we are updating a weights file,
        # or have found a bug where output did not match the upstream source.
        expected_outputs = [-1.38173, 0.16598, -2.92788, -2.66958, -0.61556]
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(outputs, expected_outputs, atol=0.01, rtol=0.01)

    def test_classifier_output(self):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "segment_ids": tf.constant([[0, 0, 0, 0]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = BertClassifier.from_preset(
            "bert_tiny_uncased_en",
        )
        # We don't assert output values, as the head weights are random.
        model(input_data)


@pytest.mark.extra_large
class BertPresetTest(tf.test.TestCase):
    """
    Test the full enumeration of our preset.

    This only tests all enumeration of our presets and is only run manually.
    Run with:
    `pytest keras_nlp/models/bert_presets_test.py --run_extra_large`
    """

    def test_load_bert(self):
        for preset in Bert.presets:
            model = Bert.from_preset(preset, load_weights=True)
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

    def test_load_bert_classifier(self):
        for preset in BertClassifier.presets:
            classifier = BertClassifier.from_preset(
                preset, num_classes=4, name="classifier"
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
            classifier(input_data)

    def test_load_preprocessors(self):
        for preset in BertPreprocessor.presets:
            preprocessor = BertPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
