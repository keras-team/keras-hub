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
"""Tests for Random Word Swap Layer."""

import tensorflow as tf

from keras_nlp.layers import random_swaps


class RandomSwapTest(tf.test.TestCase):
    def test_shape_with_scalar(self):
        augmenter = random_swaps.RandomSwaps(swaps=3)
        input = ["Running Around"]
        output = augmenter(input)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(input).shape)

    def test_get_config_and_from_config(self):

        augmenter = random_swaps.RandomSwaps(swaps=3)

        expected_config_subset = {"swaps": 3}

        config = augmenter.get_config()

        self.assertEqual(config, {**config, **expected_config_subset})

        restored_augmenter = random_swaps.RandomSwaps.from_config(
            config,
        )

        self.assertEqual(
            restored_augmenter.get_config(),
            {**config, **expected_config_subset},
        )

    def test_augment_first_batch_second(self):
        tf.random.set_seed(30)
        augmenter = random_swaps.RandomSwaps(swaps=3)

        ds = tf.data.Dataset.from_tensor_slices(
            ["samurai or ninja", "keras is good", "tensorflow is a library"]
        )
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(3))
        output = ds.take(1).get_single_element()

        exp_output = [
            b"samurai ninja or",
            b"is keras good",
            b"a is tensorflow library",
        ]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_batch_first_augment_second(self):
        tf.random.set_seed(30)
        augmenter = random_swaps.RandomSwaps(swaps=3)

        ds = tf.data.Dataset.from_tensor_slices(
            ["samurai or ninja", "keras is good", "tensorflow is a library"]
        )
        ds = ds.batch(3).map(augmenter)
        output = ds.take(1).get_single_element()

        exp_output = [
            b"samurai ninja or",
            b"is keras good",
            b"a is tensorflow library",
        ]

        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])
