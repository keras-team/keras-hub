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
"""Tests for RandomDeletion Layer."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import RandomDeletion


class RandomDeletionTest(tf.test.TestCase):
    def test_shape_and_output_from_word_deletion(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"I like", b"and Tensorflow"]
        self.assertAllEqual(output, exp_output)

    def test_shape_and_output_from_character_swaps(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.unicode_split(inputs, "UTF-8")
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey I lie", b"Keras and Tensoflow"]
        self.assertAllEqual(output, exp_output)

    def test_with_integer_tokens(self):
        keras.utils.set_random_seed(1337)
        inputs = tf.constant([[1, 2], [3, 4]])
        augmenter = RandomDeletion(rate=0.4, max_deletions=4, seed=42)
        output = augmenter(inputs)
        self.assertAllEqual(output.to_tensor().shape[0], inputs.shape[0])
        exp_output = [[2], [4]]
        self.assertAllEqual(output, exp_output)

    def test_skip_options(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomDeletion(
            rate=0.4, max_deletions=1, seed=42, skip_list=["Tensorflow", "like"]
        )
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"I like", b"and Tensorflow"]
        self.assertAllEqual(output, exp_output)

        def skip_fn(word):
            if word == "Tensorflow" or word == "like":
                return True
            return False

        augmenter = RandomDeletion(
            rate=0.4, max_deletions=1, seed=42, skip_fn=skip_fn
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey like", b"Keras Tensorflow"]
        self.assertAllEqual(output, exp_output)

        def skip_py_fn(word):
            if word == "Tensorflow" or word == "like":
                return True
            return False

        augmenter = RandomDeletion(
            rate=0.4, max_deletions=1, seed=42, skip_py_fn=skip_py_fn
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey like", b"Keras Tensorflow"]

    def test_get_config_and_from_config(self):
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)

        expected_config_subset = {"max_deletions": 1, "rate": 0.4, "seed": 42}

        config = augmenter.get_config()

        self.assertEqual(config, {**config, **expected_config_subset})

        restored_augmenter = RandomDeletion.from_config(
            config,
        )

        self.assertEqual(
            restored_augmenter.get_config(),
            {**config, **expected_config_subset},
        )

    def test_augment_first_batch_second(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()

        exp_output = [[b"I", b"like"], [b"Keras", b"and", b"Tensorflow"]]
        self.assertAllEqual(output, exp_output)

        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\pP")

        def skip_py_fn(word):
            return len(word) < 4

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_fn=skip_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [[b"I", b"like"], [b"and", b"Tensorflow"]]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_py_fn=skip_py_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [[b"Hey", b"I", b"like"], [b"and", b"Tensorflow"]]
        self.assertAllEqual(output, exp_output)

    def test_batch_first_augment_second(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()

        exp_output = [[b"I", b"like"], [b"and", b"Tensorflow"]]
        self.assertAllEqual(output, exp_output)

        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\pP")

        def skip_py_fn(word):
            return len(word) < 4

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_fn=skip_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [[b"I", b"like"], [b"and", b"Tensorflow"]]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomDeletion(
            rate=0.8, max_deletions=1, seed=42, skip_py_fn=skip_py_fn
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [[b"Hey", b"I", b"like"], [b"and", b"Tensorflow"]]
        self.assertAllEqual(output, exp_output)

    def test_functional_model(self):
        keras.utils.set_random_seed(1337)
        input_data = tf.constant(["Hey I like", "Keras and Tensorflow"])
        augmenter = RandomDeletion(rate=0.4, max_deletions=1, seed=42)
        inputs = tf.keras.Input(dtype="string", shape=())
        outputs = augmenter(tf.strings.split(inputs))
        model = tf.keras.Model(inputs, outputs)
        model_output = model(input_data)
        self.assertAllEqual(
            model_output, [[b"I", b"like"], [b"and", b"Tensorflow"]]
        )
