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
"""Tests for RandomInsertion Layer."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import RandomInsertion
from keras_nlp.tokenizers import UnicodeCodepointTokenizer


class RandomInsertionTest(tf.test.TestCase):
    def test_shape_and_output_from_word_insertion(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=["wind"]
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey wind I like", b"Keras and wind wind Tensorflow"]
        self.assertAllEqual(output, exp_output)

    def test_shape_and_output_from_character_insertion(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.unicode_split(inputs, "UTF-8")
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=["x", "y"]
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Heyxy I like", b"Keyras and Tensorfxlow"]
        self.assertAllEqual(output, exp_output)

    def test_with_integer_tokens(self):
        keras.utils.set_random_seed(1337)
        inputs = ["Hey I like", "Keras and Tensorflow"]
        tokenizer = UnicodeCodepointTokenizer(lowercase=False)
        tokenized = tokenizer.tokenize(inputs)
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=[99, 67]
        )
        augmented = augmenter(tokenized)
        output = tokenizer.detokenize(augmented)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"HeycC I like", b"KeCras and Tensorfclow"]
        self.assertAllEqual(output, exp_output)

    def test_skip_options(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=["Hey", "There"]
        )
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey Hey I like", b"Keras and There There Tensorflow"]
        self.assertAllEqual(output, exp_output)

        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\\pP")

        augmenter = RandomInsertion(
            rate=0.4,
            max_insertions=2,
            seed=42,
            insertion_list=["Hey", "There"],
            skip_fn=skip_fn,
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey Hey I like", b"Keras and There There Tensorflow"]
        self.assertAllEqual(output, exp_output)

        def skip_py_fn(word):
            if word == "Tensorflow" or word == "like":
                return True
            return False

        augmenter = RandomInsertion(
            rate=0.4,
            max_insertions=2,
            seed=42,
            insertion_list=["Hey", "There"],
            skip_py_fn=skip_py_fn,
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        output
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey Hey I like", b"Keras and There There Tensorflow"]
        self.assertAllEqual(output, exp_output)

    def test_insert_options(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=["Hey", "There"]
        )
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey Hey I like", b"Keras and There There Tensorflow"]
        self.assertAllEqual(output, exp_output)

        def insertion_fn(word):
            if word == "like":
                return "Speed"
            else:
                return "Time"

        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_fn=insertion_fn
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        output
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey Speed I like", b"Keras and Time Tensorflow Time"]
        self.assertAllEqual(output, exp_output)

        def insertion_py_fn(word):
            if len(word) > 2:
                return word[:2]
            return word

        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_py_fn=insertion_py_fn
        )
        augmented = augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        output
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b"Hey li I like", b"Keras and Ke Tensorflow Te"]
        self.assertAllEqual(output, exp_output)

    def test_get_config_and_from_config(self):
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=1, seed=42, insertion_list=["Hey", "There"]
        )

        expected_config_subset = {"max_insertions": 1, "rate": 0.4, "seed": 42}

        config = augmenter.get_config()

        self.assertEqual(config, {**config, **expected_config_subset})

        restored_augmenter = RandomInsertion.from_config(
            config,
        )

        self.assertEqual(
            restored_augmenter.get_config(),
            {**config, **expected_config_subset},
        )

    def test_augment_first_batch_second(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=["Hey", "There"]
        )
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [
            [b"Hey", b"Hey", b"I", b"like"],
            [b"Keras", b"Hey", b"and", b"Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\pP")

        def skip_py_fn(word):
            return len(word) < 4

        augmenter = RandomInsertion(
            rate=0.4,
            max_insertions=2,
            seed=42,
            insertion_list=["Hey", "There"],
            skip_fn=skip_fn,
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        exp_output = [
            [b"Hey", b"Hey", b"I", b"like"],
            [b"Keras", b"Hey", b"and", b"Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomInsertion(
            rate=0.4,
            max_insertions=2,
            seed=42,
            insertion_list=["Hey", "There"],
            skip_py_fn=skip_py_fn,
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()
        output
        exp_output = [
            [b"Hey", b"I", b"like", b"Hey"],
            [b"Keras", b"Hey", b"and", b"Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

    def test_batch_first_augment_second(self):
        keras.utils.set_random_seed(1337)
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=["Hey", "There"]
        )
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [
            [b"Hey", b"Hey", b"I", b"like"],
            [b"Keras", b"and", b"There", b"There", b"Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

        def skip_fn(word):
            return tf.strings.regex_full_match(word, r"\pP")

        def skip_py_fn(word):
            return len(word) < 4

        augmenter = RandomInsertion(
            rate=0.4,
            max_insertions=2,
            seed=42,
            insertion_list=["Hey", "There"],
            skip_fn=skip_fn,
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()
        exp_output = [
            [b"Hey", b"Hey", b"I", b"like"],
            [b"Keras", b"and", b"There", b"There", b"Tensorflow"],
        ]
        self.assertAllEqual(output, exp_output)

        augmenter = RandomInsertion(
            rate=0.4,
            max_insertions=2,
            seed=42,
            insertion_list=["Hey", "There"],
            skip_py_fn=skip_py_fn,
        )
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()
        output
        exp_output = [
            [b"Hey", b"I", b"like", b"Hey"],
            [b"Keras", b"and", b"Tensorflow", b"There", b"There"],
        ]
        self.assertAllEqual(output, exp_output)

    def test_functional_model(self):
        keras.utils.set_random_seed(1337)
        input_data = tf.constant(["Hey I like", "Keras and Tensorflow"])
        augmenter = RandomInsertion(
            rate=0.4, max_insertions=2, seed=42, insertion_list=["Hey", "There"]
        )
        inputs = tf.keras.Input(dtype="string", shape=())
        outputs = augmenter(tf.strings.split(inputs))
        model = tf.keras.Model(inputs, outputs)
        model_output = model(input_data)
        exp_output = [
            [b"Hey", b"Hey", b"I", b"like"],
            [b"Keras", b"and", b"There", b"There", b"Tensorflow"],
        ]
        self.assertAllEqual(model_output, exp_output)
