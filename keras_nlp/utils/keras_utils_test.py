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
from tensorflow import keras

from keras_nlp.utils.keras_utils import clone_initializer
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight


class CloneInitializerTest(tf.test.TestCase):
    def test_config_equality(self):
        initializer = keras.initializers.VarianceScaling(
            scale=2.0,
            mode="fan_out",
        )
        clone = clone_initializer(initializer)
        self.assertAllEqual(initializer.get_config(), clone.get_config())

    def test_random_output(self):
        initializer = keras.initializers.VarianceScaling(
            scale=2.0,
            mode="fan_out",
        )
        clone = clone_initializer(initializer)
        self.assertNotAllEqual(initializer(shape=(2, 2)), clone(shape=(2, 2)))

    def test_strings(self):
        initializer = "glorot_uniform"
        clone = clone_initializer(initializer)
        self.assertAllEqual(initializer, clone)


class PackTest(tf.test.TestCase):
    def test_pack_dict(self):
        tensor_dict = {"foo": tf.constant([1, 2])}
        data = pack_x_y_sample_weight(tensor_dict)
        self.assertAllEqual(data, tensor_dict)

    def test_pack_tuple(self):
        tensor_tuple = (tf.constant([1, 2]),)
        data = pack_x_y_sample_weight(tensor_tuple)
        self.assertAllEqual(data, (tensor_tuple,))

    def test_pack_pair(self):
        x = tf.constant([1, 2])
        y = tf.constant([3, 4])
        data = pack_x_y_sample_weight(x, y)
        self.assertAllEqual(data, (x, y))

    def test_pack_triplet(self):
        x = tf.constant([1, 2])
        y = tf.constant([3, 4])
        sw = tf.constant([5, 6])
        data = pack_x_y_sample_weight(x, y, sw)
        self.assertAllEqual(data, (x, y, sw))
