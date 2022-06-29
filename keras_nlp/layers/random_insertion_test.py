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
"""Tests for Random Word Deletion Layer."""

import tensorflow as tf

from keras_nlp.layers import random_insertion


class RandomInsertionTest(tf.test.TestCase):
    def test_shape_and_output_from_word_insertion(self):
        def replace_word(word):
            if isinstance(word, bytes):
                word = word.decode()
            dict_replacement = {"like": "admire", "bye": "ciao", "Hey": "Hi"}
            if word in dict_replacement.keys():
                return dict_replacement[word]
            return word

        inputs = ["Hey I like", "bye bye"]
        split = tf.strings.split(inputs)
        augmenter = random_insertion.RandomInsertion(
            1, 5, insertion_fn=replace_word, seed=42
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey I admire Hi like", b"ciao bye ciao bye"]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_shape_and_output_from_character_insertion(self):
        def random_chars(word):
            if isinstance(word, bytes):
                word = word.decode()
            if len(word) == 0:
                return "a"
            return word[0]

        inputs = ["Hey I like", "bye bye"]
        split = tf.strings.unicode_split(inputs, "UTF-8")
        augmenter = random_insertion.RandomInsertion(
            1, 5, insertion_fn=random_chars, seed=42
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey IlI like", b"byye ybye"]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    # def test_get_config_and_from_config(self):

    #     def random_chars(word):
    #         if isinstance(word, bytes):
    #             word = word.decode()
    #         if (len(word) == 0):
    #             return "a"
    #         return word[0]

    #     augmenter = random_insertion.RandomInsertion(1, 5, insertion_fn = random_chars, seed = 42)

    #     expected_config_subset = {
    #         'insertion_fn': <function __main__.random_chars>,
    #         'max_insertions': 5,
    #         'probability': 1,
    #         'seed': 42,
    #     }

    #     config = augmenter.get_config()

    #     self.assertEqual(config, {**config, **expected_config_subset})

    #     restored_augmenter = (
    #         random_insertion.RandomInsertion.from_config(
    #             config,
    #         )
    #     )

    #     self.assertEqual(
    #         restored_augmenter.get_config(),
    #         {**config, **expected_config_subset},
    #     )
