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
"""Tests for Roberta model."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models import roberta


class RobertaTest(tf.test.TestCase):
    def test_valid_call_roberta(self):
        model = roberta.RobertaCustom(
            vocabulary_size=50265,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=12,
            name="encoder",
        )
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 12), dtype=tf.int64, maxval=50265
            ),
            "input_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }
        model(input_data)

    def test_valid_call_classifier(self):
        model = roberta.RobertaCustom(
            vocabulary_size=50265,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=12,
            name="encoder",
        )
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 12), dtype=tf.int64, maxval=50265
            ),
            "input_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }
        classifier = roberta.RobertaClassifier(model, 4, 768, name="classifier")
        classifier(input_data)

    def test_valid_call_roberta_base(self):
        model = roberta.RobertaBase(name="encoder")
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
            ),
            "input_mask": tf.constant([1] * 512, shape=(1, 512)),
        }
        model(input_data)

    def test_variable_sequence_length_call_roberta(self):
        model = roberta.RobertaCustom(
            vocabulary_size=50265,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=100,
            name="encoder",
        )
        for seq_length in (25, 50, 100):
            input_data = {
                "input_ids": tf.ones((8, seq_length), dtype="int32"),
                "input_mask": tf.ones((8, seq_length), dtype="int32"),
            }
            output = model(input_data)
            self.assertAllEqual(
                tf.shape(output["sequence_output"]), [8, seq_length, 768]
            )

    def test_saving_model(self):
        model = roberta.RobertaCustom(
            vocabulary_size=50265,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=12,
            name="encoder",
        )
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 12), dtype=tf.int64, maxval=50265
            ),
            "input_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }
        model_output = model.predict(input_data)

        save_path = os.path.join(self.get_temp_dir(), "model")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        restored_output = restored_model.predict(input_data)
        self.assertAllClose(
            model_output["sequence_output"],
            restored_output["sequence_output"],
        )
