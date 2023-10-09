# Copyright 2023 The KerasNLP Authors
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

import keras_nlp.layers.modeling.transformer_layer_utils as utils
from keras_nlp.backend import ops
from keras_nlp.backend import random
from keras_nlp.tests.test_case import TestCase


class TransformerLayerUtilsTest(TestCase):
    def test_compute_causal_mask(self):
        mask = utils.compute_causal_mask(1, 2, 2)
        self.assertAllEqual(mask, [[[1, 0], [1, 1]]])

    def test_merge_padding_and_attention_mask(self):
        padding_mask = ops.array([[1, 1, 0]])
        attention_mask = ops.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
        inputs = random.uniform(shape=[1, 3, 2])
        merged_mask = utils.merge_padding_and_attention_mask(
            inputs,
            padding_mask,
            attention_mask,
        )
        self.assertAllEqual(merged_mask, [[[0, 0, 0], [0, 1, 0], [1, 0, 0]]])

    def test_bad_mask_shapes(self):
        with self.assertRaises(ValueError):
            padding_mask = ops.array([[[1, 1, 0], [1, 0, 0]]])
            attention_mask = ops.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            inputs = random.uniform(shape=[1, 3, 2])
            utils.merge_padding_and_attention_mask(
                inputs,
                padding_mask,
                attention_mask,
            )

        with self.assertRaises(ValueError):
            padding_mask = ops.array([[1, 1, 0]])
            attention_mask = ops.array([[0, 0, 1], [1, 0, 0]])
            inputs = random.uniform(shape=[1, 3, 2])
            utils.merge_padding_and_attention_mask(
                inputs,
                padding_mask,
                attention_mask,
            )
