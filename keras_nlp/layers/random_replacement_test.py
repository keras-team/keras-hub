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
from tensorflow import keras

from keras_nlp.layers import random_replacement


class RandomReplacementTest(tf.test.TestCase):
    def test_shape_and_output_from_word_replacement(self):
        keras.utils.set_random_seed(1337)
        inputs=["Hey I like", "Keras and Tensorflow"]
        split=tf.strings.split(inputs)
        augmenter=random_replacement.RandomReplacement(rate=0.3, 
        max_replacements = 2, seed=42, replacement_list = 
        ['Random1', 'Random2', 'Random3'])
        augmented=augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b'Random1 Random1 like', b'Random3 Random3 Tensorflow']
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_shape_and_output_from_character_replacement(self):
        keras.utils.set_random_seed(1337)
        inputs=["Hey I like", "Keras and Tensorflow"]
        split=tf.strings.unicode_split(inputs, 'UTF-8')
        augmenter=random_replacement.RandomReplacement(rate=0.3, 
        max_replacements = 2, seed=42, replacement_list = ['x', 'y', 'z'])
        augmented=augmenter(split)
        output = tf.strings.reduce_join(augmented, axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b'xxy I like', b'yyras and Tensorflow']
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_skip_options(self):
        def py_replacement_fn(word):
            if len(word) < 4:
              return word[:2]
            return word[:4]
        keras.utils.set_random_seed(23)
        inputs=["Hey I like", "Keras and Tensorflow and Food"]
        split=tf.strings.split(inputs)
        augmenter=random_replacement.RandomReplacement(rate=0.5, max_replacements=6, seed=42,
        py_replacement_fn=py_replacement_fn, skip_list=['Keras'])
        augmented=augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b'He I like', b'Keras an Tensorflow and Food']
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

        def skip_fn(word):
            if word == 'Hey':
                return True
            return False
        keras.utils.set_random_seed(1337)
        augmenter=random_replacement.RandomReplacement(rate=0.5, max_replacements=6, seed=42,
        py_replacement_fn=py_replacement_fn, skip_fn=skip_fn)
        augmented=augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b'Hey I like', b'Kera an Tensorflow and Food']
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

        def py_skip_fn(word):
            if len(word) == 3:
                return True
            return False
        keras.utils.set_random_seed(1337)
        augmenter=random_replacement.RandomReplacement(rate=0.5, max_replacements=6, seed=42,
        py_replacement_fn=py_replacement_fn, py_skip_fn=py_skip_fn)
        augmented=augmenter(split)
        output = tf.strings.reduce_join(augmented, separator=" ", axis=-1)
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)
        exp_output = [b'Hey I like', b'Kera and Tensorflow and Food']
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])
        self.assertAllEqual(output.shape, tf.convert_to_tensor(inputs).shape)

    def test_augment_first_batch_second(self):
        keras.utils.set_random_seed(1337)
        augmenter=random_replacement.RandomReplacement(rate=0.3, 
        max_replacements = 2, seed=42, replacement_list = 
        ['Random1', 'Random2', 'Random3'])
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.map(augmenter)
        ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(2))
        output = ds.take(1).get_single_element()

        exp_output = [[b'Random1', b'Random1', b'like'],
                    [b'Random3', b'Random3', b'Tensorflow']]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_batch_first_augment_second(self):
        keras.utils.set_random_seed(1337)
        augmenter=random_replacement.RandomReplacement(rate=0.3, 
        max_replacements = 2, seed=42, replacement_list = 
        ['Random1', 'Random2', 'Random3'])
        inputs = ["Hey I like", "Keras and Tensorflow"]
        split = tf.strings.split(inputs)
        ds = tf.data.Dataset.from_tensor_slices(split)
        ds = ds.batch(5).map(augmenter)
        output = ds.take(1).get_single_element()

        exp_output =  [[b'Random1', b'Random1', b'like'],
                    [b'Random3', b'Random3', b'Tensorflow']]
        for i in range(output.shape[0]):
            self.assertAllEqual(output[i], exp_output[i])

    def test_functional_model(self):
        keras.utils.set_random_seed(1337)
        input_data = tf.constant(["Hey I like", "Keras and Tensorflow"])
        augmenter=random_replacement.RandomReplacement(rate=0.3, 
        max_replacements = 2, seed=42, replacement_list = 
        ['Random1', 'Random2', 'Random3'])
        inputs = tf.keras.Input(dtype="string", shape=())
        outputs = augmenter(tf.strings.split(inputs))
        model = tf.keras.Model(inputs, outputs)
        model_output = model(input_data)
        self.assertAllEqual(
            model_output,  [[b'Random1', b'Random1', b'like'], 
                            [b'Random3', b'Random3', b'Tensorflow']]
        )