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
from absl import parameterized

from keras_nlp.models.bert import bert_checkpoints
from keras_nlp.models.bert import bert_models
from keras_nlp.models.bert import bert_preprocessing
from keras_nlp.models.bert import bert_tasks
from keras_nlp.models.gpt2 import checkpoints as gpt2_checkpoints
from keras_nlp.models.gpt2 import model_classes as gpt2_models


@pytest.mark.slow
class BertCkptTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            checkpoint,
            checkpoint,
            bert_models.model_class_by_name(
                bert_checkpoints.checkpoints[checkpoint]["model"]
            ),
        )
        for checkpoint in bert_checkpoints.checkpoints
    )
    def test_load_bert(self, checkpoint, bert_class):
        model = bert_class(weights=checkpoint)
        input_data = {
            "token_ids": tf.random.uniform(
                shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
            ),
            "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
            "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
        }
        model(input_data)

    @parameterized.named_parameters(
        (checkpoint, checkpoint) for checkpoint in bert_checkpoints.checkpoints
    )
    def test_load_bert_backbone_string(self, checkpoint):
        classifier = bert_tasks.BertClassifier(checkpoint, 4, name="classifier")
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

    @parameterized.named_parameters(
        (vocabulary, vocabulary) for vocabulary in bert_checkpoints.vocabularies
    )
    def test_load_vocabularies(self, vocabulary):
        tokenizer = bert_preprocessing.BertPreprocessor(
            vocabulary=vocabulary,
        )
        tokenizer("The quick brown fox.")


@pytest.mark.slow
class Gpt2CkptTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            checkpoint,
            checkpoint,
            gpt2_models.model_class_by_name(
                gpt2_checkpoints[checkpoint]["model"]
            ),
        )
        for checkpoint in gpt2_checkpoints.checkpoints
    )
    def test_load(self, checkpoint, gpt2_class):
        model = gpt2_class(weights=checkpoint)
        input_data = {
            "token_ids": tf.random.uniform(
                shape=(1, 1024),
                dtype=tf.int64,
                maxval=model.vocabulary_size,
            ),
            "padding_mask": tf.constant([1] * 1024, shape=(1, 1024)),
        }
        model(input_data)

    @parameterized.named_parameters(
        (model_class, model_class)
        for model_class in set(
            [
                gpt2_checkpoints[checkpoint]["model"]
                for checkpoint in gpt2_checkpoints.checkpoints
            ]
        )
    )
    def test_defaults(self, model_class):
        model_class()
