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

import pytest

from keras_nlp.backend import keras
from keras_nlp.backend import random
from keras_nlp.layers.modeling.lora_dense import LoraDense
from keras_nlp.tests.test_case import TestCase


@pytest.mark.multi_backend_only
class LoraDenseTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=LoraDense,
            init_kwargs={
                "inner_dense": keras.layers.Dense(16),
                "rank": 2,
                "alpha": 16,
                "lora_a_initializer": "HeNormal",
            },
            input_data=random.uniform(shape=(2, 4, 8)),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=2,
            expected_num_non_trainable_variables=2,
            run_mixed_precision_check=False,
        )

    def test_layer_behaviors_einsum(self):
        self.run_layer_test(
            cls=LoraDense,
            init_kwargs={
                "inner_dense": keras.layers.EinsumDense(
                    "abc,cde->abde",
                    output_shape=(None, 2, 16),
                ),
                "lora_a_initializer": "HeNormal",
            },
            input_data=random.uniform(shape=(2, 4, 8)),
            expected_output_shape=(2, 4, 2, 16),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=1,
            expected_num_non_trainable_variables=1,
            run_mixed_precision_check=False,
        )

    def test_merge_dense(self):
        inner_dense = keras.layers.Dense(16)
        layer = LoraDense(inner_dense, rank=4)
        layer.build((2, 16))
        layer.lora_a.assign(random.uniform(shape=(16, 4)))
        layer.lora_b.assign(random.uniform(shape=(4, 16)))

        input_data = random.uniform((2, 16))
        lora_output = layer(input_data)
        dense_output = inner_dense(input_data)
        self.assertNotAllClose(lora_output, dense_output)

        layer.merge_weights()
        merged_lora_output = layer(input_data)
        dense_output = inner_dense(input_data)
        self.assertAllClose(lora_output, merged_lora_output)
        self.assertAllClose(lora_output, dense_output)

    def test_merge_einsum(self):
        inner_dense = keras.layers.EinsumDense(
            "abc,cde->abde",
            output_shape=(None, 2, 16),
        )
        layer = LoraDense(inner_dense, rank=4)
        layer.build((2, 4, 16))
        layer.lora_a.assign(random.uniform(shape=(16, 4)))
        layer.lora_b.assign(random.uniform(shape=(4, 2, 16)))

        input_data = random.uniform((2, 4, 16))
        lora_output = layer(input_data)
        dense_output = inner_dense(input_data)
        self.assertNotAllClose(lora_output, dense_output)

        layer.merge_weights()
        merged_lora_output = layer(input_data)
        dense_output = inner_dense(input_data)
        self.assertAllClose(lora_output, merged_lora_output)
        self.assertAllClose(lora_output, dense_output)

    def test_freezing(self):
        inner_dense = keras.layers.Dense(16)
        layer = LoraDense(inner_dense, freeze_bias=False)
        layer.build((2, 16))
        self.assertFalse(inner_dense.kernel.trainable)
        self.assertTrue(inner_dense.bias.trainable)

        inner_dense = keras.layers.Dense(16)
        layer = LoraDense(inner_dense)
        layer.build((2, 16))
        self.assertFalse(inner_dense.kernel.trainable)
        self.assertFalse(inner_dense.bias.trainable)

    def test_errors_if_not_dense(self):
        with self.assertRaises(ValueError):
            LoraDense(keras.layers.Concatenate())

    def test_errors_invalid_einsum(self):
        with self.assertRaises(ValueError):
            # Kernel feature dim in the wrong place.
            einsum = keras.layers.EinsumDense("abc,dec->abde", (2, 2, 16))
            LoraDense(einsum, rank=4)

        with self.assertRaises(ValueError):
            # Input feature dim in the wrong place.
            einsum = keras.layers.EinsumDense("acb,cde->abde", (2, 2, 16))
            LoraDense(einsum, rank=4)

        with self.assertRaises(ValueError):
            # Input feature dim not summed over.
            einsum = keras.layers.EinsumDense("abc,cde->abcde", (2, 2, 2, 16))
            LoraDense(einsum, rank=4)

        with self.assertRaises(ValueError):
            # Double summations.
            einsum = keras.layers.EinsumDense("abcd,cde->abe", (2, 2, 16))
            LoraDense(einsum, rank=4)
