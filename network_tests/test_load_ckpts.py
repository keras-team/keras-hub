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

import tensorflow as tf

import keras_nlp


class BertCkptTest(tf.test.TestCase):
    def test_load_bert_base_uncased(self):
        model = keras_nlp.models.BertBase(weights="uncased_en")
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
            ),
            "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
            "input_mask": tf.constant([1] * 512, shape=(1, 512)),
        }
        model(input_data)

    def test_load_bert_base_cased(self):
        model = keras_nlp.models.BertBase(weights="cased_en")
        input_data = {
            "input_ids": tf.random.uniform(
                shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
            ),
            "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
            "input_mask": tf.constant([1] * 512, shape=(1, 512)),
        }
        model(input_data)
