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
"""Tests for Transformer Decoder."""

import tensorflow as tf

from keras_nlp.layers import random_deletion


class RandomDeletionTest(tf.test.TestCase):
    def test_shape_with_scalar(self):
        augmenter = random_deletion.RandomDeletion(
            probability=0.5, max_deletions=3
        )
        input = ["Running Around"]
        output = augmenter(input)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(input).shape)

    def test_shape_with_nested(self):
        augmenter = random_deletion.RandomDeletion(
            probability=0.5, max_deletions=3
        )
        input = [
            ["dog dog dog dog dog", "I Like CATS"],
            ["I Like to read comics", "Shinobis and Samurais"],
        ]
        output = augmenter(input)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(input).shape)

    def test_get_config_and_from_config(self):
        augmenter = random_deletion.RandomDeletion(
            probability=0.5, max_deletions=3
        )

        config = augmenter.get_config()

        expected_config_subset = {
            "probability": 0.5,
            "max_deletions": 3,
        }

        self.assertEqual(config, {**config, **expected_config_subset})

        restored_augmenter = random_deletion.RandomDeletion.from_config(
            config,
        )

        self.assertEqual(
            restored_augmenter.get_config(),
            {**config, **expected_config_subset},
        )
