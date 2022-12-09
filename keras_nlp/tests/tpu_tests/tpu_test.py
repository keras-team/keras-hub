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


import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.models.bert.bert_backbone import BertBackbone


class BertBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 8

        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
            tpu="local"
        )
        self.strategy = tf.distribute.TPUStrategy(resolver)
        with self.strategy.scope():
            self.model = BertBackbone(
                vocabulary_size=1000,
                num_layers=2,
                num_heads=2,
                hidden_dim=64,
                intermediate_dim=128,
                max_sequence_length=128,
            )

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
        with self.strategy.scope():
            print(self.strategy)
            self.model(self.input_batch)
        # Check default name passed through
        self.assertEqual(self.model.name, "backbone")
