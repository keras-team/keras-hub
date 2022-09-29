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
"""Tests for RoBERTa task specific models and heads."""

import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.models.roberta.roberta_models import RobertaCustom
from keras_nlp.models.roberta.roberta_tasks import RobertaClassifier


class RobertaClassifierTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.model = RobertaCustom(
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
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_classifier(self):
        classifier = RobertaClassifier(self.model, 4, 128, name="classifier")
        classifier(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_classifier_compile(self, jit_compile):
        model = RobertaClassifier(self.model, 4, 128, name="classifier")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_roberta_classifier_compile_batched_ds(self, jit_compile):
        model = RobertaClassifier(self.model, 4, 128, name="classifier")
        model.compile(jit_compile=jit_compile)
        model.predict(self.input_dataset)
