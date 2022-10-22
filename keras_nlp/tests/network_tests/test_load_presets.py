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


@pytest.mark.slow
class BertPresetTest(tf.test.TestCase):
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

    def test_load_bert_backbone_string(self):
        for preset in Bert.presets:
            classifier = BertClassifier(preset, 4, name="classifier")
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

    def test_classifier_default_args(self):
        classifier = BertClassifier()
        input_data = {
            "token_ids": tf.random.uniform(
                shape=(1, 512),
                dtype=tf.int64,
                maxval=classifier.backbone.vocabulary_size,
            ),
            "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
            "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
        }
        classifier(input_data)

    def test_load_preprocessors(self):
        for preset in BertPreprocessor.presets:
            preprocessor = BertPreprocessor.from_preset(preset)
            preprocessor("The quick brown fox.")
