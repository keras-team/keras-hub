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
"""Test for DeBERTa backbone models."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.deberta_v3.deberta_v3_backbone import DebertaV3Backbone


class DebertaV3BackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.model = DebertaV3Backbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            bucket_size=64,
        )
        self.batch_size = 8
        self.input_batch = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_deberta(self):
        self.model(self.input_batch)

        # Check default name passed through
        self.assertRegexpMatches(self.model.name, "deberta_v3_backbone")

    def test_variable_sequence_length_call_deberta(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            self.model(input_data)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compile(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compile_batched_ds(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_dataset)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model_output = self.model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, DebertaV3Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)
