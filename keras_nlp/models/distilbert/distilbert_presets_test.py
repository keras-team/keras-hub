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

from keras_nlp.models.distilbert.distilbert_models import DistilBert
from keras_nlp.models.distilbert.distilbert_preprocessing import (
    DistilBertPreprocessor,
)
from keras_nlp.models.distilbert.distilbert_preprocessing import (
    DistilBertTokenizer,
)
from keras_nlp.models.distilbert.distilbert_tasks import DistilBertClassifier


@pytest.mark.large
class DistilBertPresetSmokeTest(tf.test.TestCase):
    """
    A smoke test for DistilBERT presets we run continuously.

    This only tests the smallest weights we have available. Run with:
    `pytest keras_nlp/models/distilbert/distilbert_presets_test.py --run_large`
    """

    def test_tokenizer_output(self):
        tokenizer = DistilBertTokenizer.from_preset(
            "distilbert_base_uncased_en",
        )
        outputs = tokenizer("The quick brown fox.")
        expected_outputs = [1996, 4248, 2829, 4419, 1012]
        self.assertAllEqual(outputs, expected_outputs)

    def test_preprocessor_output(self):
        tokenizer = DistilBertPreprocessor.from_preset(
            "distilbert_base_uncased_en",
            sequence_length=4,
        )
        outputs = tokenizer("The quick brown fox.")["token_ids"]
        expected_outputs = [101, 1996, 4248, 102]
        self.assertAllEqual(outputs, expected_outputs)

    def test_backbone_output(self):
        input_data = {
            "token_ids": tf.constant([[101, 1996, 4248, 102]]),
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = DistilBert.from_preset(
            "distilbert_base_uncased_en",
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
            "padding_mask": tf.constant([[1, 1, 1, 1]]),
        }
        model = DistilBertClassifier.from_preset(
            "distilbert_base_uncased_en",
        )
        # We don't assert output values, as the head weights are random.
        model(input_data)


@pytest.mark.extra_large
class DistilBertPresetTest(tf.test.TestCase):
    """
    Test the full enumeration of our preset.

    This only tests all enumeration of our presets and is only run manually.
    Run with:
    `pytest keras_nlp/models/distilbert_presets_test.py --run_extra_large`
    """

    def test_load_distilbert(self):
        for preset in DistilBert.presets:
            model = DistilBert.from_preset(preset, load_weights=True)
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            model(input_data)

    def test_load_distilbert_classifier(self):
        for preset in DistilBertClassifier.presets:
            classifier = DistilBertClassifier.from_preset(
                preset, num_classes=4, name="classifier"
            )
            input_data = {
                "token_ids": tf.random.uniform(
                    shape=(1, 512),
                    dtype=tf.int64,
                    maxval=classifier.backbone.vocabulary_size,
                ),
                "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
            }
            classifier(input_data)

    def test_load_tokenizers(self):
        for preset in DistilBertTokenizer.presets:
            tokenizer = DistilBertTokenizer.from_preset(preset)
            tokenizer("The quick brown fox.")

    def test_load_preprocessors(self):
        for preset in DistilBertPreprocessor.presets:
            preprocessor = DistilBertPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
