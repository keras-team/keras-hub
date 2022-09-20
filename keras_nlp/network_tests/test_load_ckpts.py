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
"""Tests for loading pretrained model checkpoints."""

import pytest
import tensorflow as tf

import keras_nlp
from keras_nlp.models.bert import checkpoints as bert_checkpoints
from keras_nlp.models.bert import (
    model_class_by_name as bert_model_class_by_name,
)
from keras_nlp.models.bert import vocabularies as bert_vocabularies


@pytest.mark.slow
class BertCkptTest(tf.test.TestCase):
    def test_load_bert(self):
        for checkpoint in bert_checkpoints:
            bert_variant = bert_checkpoints[checkpoint]["model"]
            bert_class = bert_model_class_by_name(bert_variant)
            model = bert_class(weights=checkpoint)
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
        for checkpoint in bert_checkpoints:
            classifier = keras_nlp.models.bert_tasks.BertClassifier(
                checkpoint, 4, name="classifier"
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

    def test_classifier_default_args(self):
        classifier = keras_nlp.models.bert_tasks.BertClassifier()
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
        for vocabulary in bert_vocabularies:
            tokenizer = keras_nlp.models.BertPreprocessor(vocabulary=vocabulary)
            tokenizer("The quick brown fox.")
