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

import tensorflow as tf

import keras_nlp.layers.transformer_layer_utils as utils


class TransformerEncoderTest(tf.test.TestCase):
    def test_compute_causal_mask(self):
        inputs = tf.random.uniform(shape=[1, 2, 2])
        mask = utils.compute_causal_mask(inputs)
        self.assertTrue((mask.numpy() == [[1, 0], [1, 1]]).all())

    def test_merge_padding_and_attention_mask(self):
        padding_mask = tf.convert_to_tensor([[1, 1, 0]])
        attention_mask = tf.convert_to_tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        inputs = tf.random.uniform(shape=[1, 3, 2])
        merged_mask = utils.merge_padding_and_attention_mask(
            inputs,
            padding_mask,
            attention_mask,
        )
        self.assertTrue(
            (merged_mask.numpy() == [[0, 0, 0], [0, 1, 0], [1, 0, 0]]).all()
        )
