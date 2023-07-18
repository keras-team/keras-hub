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
import tensorflow as tf
from absl.testing import parameterized

from keras_nlp.backend import ops


def convert_to_comparible_type(x):
    """Convert tensors to comparable types.

    Any string are converted to plain python types. Any jax or torch tensors
    are converted to numpy.
    """
    if getattr(x, "dtype", None) == tf.string:
        if isinstance(x, tf.RaggedTensor):
            x = x.to_list()
        if isinstance(x, tf.Tensor):
            x = x.numpy() if x.shape.rank == 0 else x.numpy().tolist()
        return tf.nest.map_structure(lambda x: x.decode("utf-8"), x)
    if isinstance(x, (tf.Tensor, tf.RaggedTensor)):
        return x
    if ops.is_tensor(x):
        return ops.convert_to_numpy(x)
    return x


class TestCase(tf.test.TestCase, parameterized.TestCase):
    """Base test case class for KerasNLP."""

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        x1 = tf.nest.map_structure(convert_to_comparible_type, x1)
        x2 = tf.nest.map_structure(convert_to_comparible_type, x2)
        super().assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)

    def assertEqual(self, x1, x2, msg=None):
        x1 = tf.nest.map_structure(convert_to_comparible_type, x1)
        x2 = tf.nest.map_structure(convert_to_comparible_type, x2)
        super().assertEqual(x1, x2, msg=msg)

    def assertAllEqual(self, x1, x2, msg=None):
        x1 = tf.nest.map_structure(convert_to_comparible_type, x1)
        x2 = tf.nest.map_structure(convert_to_comparible_type, x2)
        super().assertAllEqual(x1, x2, msg=msg)
