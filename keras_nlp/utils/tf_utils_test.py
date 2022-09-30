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

from keras_nlp.utils.tf_utils import tensor_to_list
from keras_nlp.utils.tf_utils import tensor_to_string_list


class TensorToListTest(tf.test.TestCase):
    def test_ragged_input(self):
        input_data = tf.ragged.constant([[1, 2], [4, 5, 6]])
        list_output = tensor_to_list(input_data)
        self.assertAllEqual(list_output, [[1, 2], [4, 5, 6]])

    def test_dense_input(self):
        input_data = tf.constant([[1, 2], [3, 4]])
        list_output = tensor_to_list(input_data)
        self.assertAllEqual(list_output, [[1, 2], [3, 4]])

    def test_scalar_input(self):
        input_data = tf.constant(1)
        list_output = tensor_to_list(input_data)
        self.assertEqual(list_output, 1)


class TensorToStringListTest(tf.test.TestCase):
    def test_detokenize_to_strings_for_ragged(self):
        input_data = tf.ragged.constant([["▀▁▂▃", "samurai"]])
        detokenize_output = tensor_to_string_list(input_data)
        self.assertAllEqual(detokenize_output, [["▀▁▂▃", "samurai"]])

    def test_detokenize_to_strings_for_dense(self):
        input_data = tf.constant([["▀▁▂▃", "samurai"]])
        detokenize_output = tensor_to_string_list(input_data)
        self.assertAllEqual(detokenize_output, [["▀▁▂▃", "samurai"]])

    def test_detokenize_to_strings_for_scalar(self):
        input_data = tf.constant("▀▁▂▃")
        detokenize_output = tensor_to_string_list(input_data)
        self.assertEqual(detokenize_output, "▀▁▂▃")
