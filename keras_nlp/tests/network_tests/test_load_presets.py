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

from keras_nlp.models.bert import bert_models
from keras_nlp.models.bert import bert_preprocessing
from keras_nlp.models.bert import bert_presets
from keras_nlp.models.bert import bert_tasks


@pytest.mark.slow
class BertPresetTest(tf.test.TestCase):
    def test_load_bert(self):
        for preset in bert_models.Bert.presets:
            model = bert_models.Bert.from_preset(preset, load_weights=True)
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
        for preset in bert_models.Bert.presets:
            classifier = bert_tasks.BertClassifier(preset, 4, name="classifier")
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
        classifier = bert_tasks.BertClassifier()
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

    def test_load_vocabularies(self):
        for vocabulary in bert_presets.vocabularies:
            tokenizer = bert_preprocessing.BertPreprocessor(
                vocabulary=vocabulary,
            )
            tokenizer("The quick brown fox.")
