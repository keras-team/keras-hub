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
"""Tests for BERT task specific models and heads."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.bert.bert_models import Bert
from keras_nlp.models.bert.bert_tasks import BertClassifier


class BertClassifierTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.backbone = Bert(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
            name="encoder",
        )
        self.classifier = BertClassifier(self.backbone, 4, name="classifier")
        self.batch_size = 8
        self.input_batch = {
            "token_ids": tf.ones(
                (self.batch_size, self.backbone.max_sequence_length),
                dtype="int32",
            ),
            "segment_ids": tf.ones(
                (self.batch_size, self.backbone.max_sequence_length),
                dtype="int32",
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.backbone.max_sequence_length),
                dtype="int32",
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_classifier(self):
        self.classifier(self.input_batch)

    def test_valid_call_presets(self):
        # Test preset loading without weights
        for preset in BertClassifier.presets:
            classifier = BertClassifier.from_preset(preset, load_weights=False)
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, self.backbone.max_sequence_length),
                    dtype="int32",
                ),
                "segment_ids": tf.ones(
                    (self.batch_size, self.backbone.max_sequence_length),
                    dtype="int32",
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, self.backbone.max_sequence_length),
                    dtype="int32",
                ),
            }
            classifier(input_data)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            BertClassifier.from_preset(
                "bert_base_uncased_clowntown",
                load_weights=False,
            )

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_compile(self, jit_compile):
        self.classifier.compile(jit_compile=jit_compile)
        self.classifier.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_bert_classifier_compile_batched_ds(self, jit_compile):
        self.classifier.compile(jit_compile=jit_compile)
        self.classifier.predict(self.input_dataset)

    @parameterized.named_parameters(
        ("save_format_tf", "tf"), ("save_format_h5", "h5")
    )
    def test_saving_model(self, save_format):
        model_output = self.classifier(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "model")
        self.classifier.save(save_path, save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, BertClassifier)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)
