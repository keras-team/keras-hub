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
"""Tests for Bert model."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models import bert


class BertTest(tf.test.TestCase):
    def test_valid_call_bert(self):
        model = bert.Bert(
            vocabulary_size=30522,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=12,
            name="encoder",
        )
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 12), dtype=tf.int64, maxval=30522
            ),
            "segment_ids": tf.constant(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
            "input_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }
        model(input_data)

    def test_valid_call_classifier(self):
        model = bert.Bert(
            vocabulary_size=30522,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=12,
            name="encoder",
        )
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 12), dtype=tf.int64, maxval=30522
            ),
            "segment_ids": tf.constant(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
            "input_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }
        classifier = bert.BertClassifier(model, 4, name="classifier")
        classifier(input_data)

    def test_valid_call_bert_base(self):
        model = bert.BertBase(name="encoder")
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
            ),
            "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
            "input_mask": tf.constant([1] * 512, shape=(1, 512)),
        }
        model(input_data)

    def test_saving_model(self):
        model = bert.Bert(
            vocabulary_size=30522,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=12,
            name="encoder",
        )
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 12), dtype=tf.int64, maxval=30522
            ),
            "segment_ids": tf.constant(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
            "input_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }
        model_output = model(input_data)
        save_path = os.path.join(self.get_temp_dir(), "model")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        restored_output = restored_model(input_data)
        self.assertAllClose(
            model_output["pooled_output"], restored_output["pooled_output"]
        )
