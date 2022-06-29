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
"""Tests for Random Swap Layer."""

import tensorflow as tf

from keras_nlp.layers import random_swaps


class RandomSwapTest(tf.test.TestCase):
    def test_shape_and_output_from_word_swaps(self):
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmenter = random_swaps.RandomSwaps(3, seed=42)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b'I like Hey', b'and Tensorflow Keras']
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_shape_and_output_from_character_swaps(self):
        inputs = ["Hey I like", "bye bye"]
        split = tf.strings.unicode_split(inputs, "UTF-8")
        augmenter = random_swaps.RandomSwaps(1, seed=42)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b'HeI y like', b'b eybye']
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_get_config_and_from_config(self):

        augmenter = random_swaps.RandomSwaps(1, seed=42)

        expected_config_subset = {
            'seed': 42,
            'swaps': 1,
        }

        config = augmenter.get_config()

        self.assertEqual(config, {**config, **expected_config_subset})

        restored_augmenter = (
            random_swaps.RandomSwaps.from_config(
                config,
            )
        )

        self.assertEqual(
            restored_augmenter.get_config(),
            {**config, **expected_config_subset},
        )
