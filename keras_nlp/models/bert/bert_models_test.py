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
"""Test for BERT backbone models."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.bert.bert_models import Bert


class BertTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.model = Bert(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            name="encoder",
        )
        self.batch_size = 8
        self.input_batch = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "segment_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_bert(self):
        self.model(self.input_batch)
        self.assertEqual(self.model.name, "encoder")

    def test_variable_sequence_length_call_bert(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "segment_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            self.model(input_data)

    def test_valid_call_presets(self):
        # Test preset loading without weights
        for preset in Bert.presets:
            model = Bert.from_preset(preset, load_weights=False, name="encoder")
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, self.model.max_sequence_length),
                    dtype="int32",
                ),
                "segment_ids": tf.ones(
                    (self.batch_size, self.model.max_sequence_length),
                    dtype="int32",
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, self.model.max_sequence_length),
                    dtype="int32",
                ),
            }
            model(input_data)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            Bert.from_preset(
                "bert_base_uncased_clowntown",
                load_weights=False,
                name="encoder",
            )

    def test_preset_mutability(self):
        preset = "bert_base_uncased_en"
        # Cannot overwrite the presents attribute in an object
        with self.assertRaises(AttributeError):
            self.model.presets = {"my_model": "clowntown"}
        # Cannot mutate presents in an object
        config = self.model.presets[preset]["config"]
        config["max_sequence_length"] = 1
        self.assertEqual(config["max_sequence_length"], 1)
        self.assertEqual(
            self.model.presets[preset]["config"]["max_sequence_length"], 512
        )
        # Cannot mutate presets in the class
        config = Bert.presets[preset]["config"]
        config["max_sequence_length"] = 1
        self.assertEqual(config["max_sequence_length"], 1)
        self.assertEqual(
            Bert.presets[preset]["config"]["max_sequence_length"], 512
        )

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
        ("save_format_tf", "tf"), ("save_format_h5", "h5")
    )
    def test_saving_model(self, save_format):
        model_output = self.model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "model")
        self.model.save(save_path, save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, Bert)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            model_output["pooled_output"], restored_output["pooled_output"]
        )
