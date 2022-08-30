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
    def setUp(self):
        self.model = roberta.RobertaCustom(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            name="encoder",
        )
        self.batch_size = 8
        self.input_data = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

    def test_valid_call_roberta(self):
        self.model(self.input_data)

    def test_valid_call_classifier(self):
        classifier = roberta.RobertaClassifier(
            self.model, 4, 128, name="classifier"
        )
        classifier(self.input_data)

    def test_valid_call_roberta_base(self):
        model = roberta.RobertaBase(vocabulary_size=1000, name="encoder")
        input_data = {
            "token_ids": tf.ones(
                (self.batch_size, model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, model.max_sequence_length), dtype="int32"
            ),
        }
        model(input_data)

    def test_variable_sequence_length_call_roberta(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            output = self.model(input_data)
            self.assertAllEqual(
                tf.shape(output["sequence_output"]),
                [self.batch_size, seq_length, self.model.hidden_dim],
            )

    def test_saving_model(self):
        model_output = self.model.predict(self.input_data)

        save_path = os.path.join(self.get_temp_dir(), "model")
        self.model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        restored_output = restored_model.predict(self.input_data)
        self.assertAllClose(
            model_output["sequence_output"],
            restored_output["sequence_output"],
        )
