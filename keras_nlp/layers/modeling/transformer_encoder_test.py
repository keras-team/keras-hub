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

from absl.testing import parameterized

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.backend import random
from keras_nlp.layers.modeling.transformer_encoder import TransformerEncoder
from keras_nlp.tests.test_case import TestCase


class TransformerEncoderTest(TestCase):
    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_layer_behaviors(self, normalize_first):
        self.run_layer_test(
            cls=TransformerEncoder,
            init_kwargs={
                "intermediate_dim": 4,
                "num_heads": 2,
                "normalize_first": normalize_first,
                "activation": "relu",
                "layer_norm_epsilon": 1e-05,
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
                "dropout": 0.1,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
            expected_num_trainable_weights=16,
            expected_num_non_trainable_variables=3,  # dropout rng seeds
        )

    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_valid_call(self, normalize_first):
        encoder = TransformerEncoder(
            intermediate_dim=4,
            num_heads=2,
            normalize_first=normalize_first,
        )
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                encoder,
            ]
        )
        input = random.uniform(shape=[2, 4, 6])
        model(input)

    def test_valid_call_with_mask(self):
        encoder = TransformerEncoder(
            intermediate_dim=4,
            num_heads=2,
        )
        encoder.build([2, 4, 6])
        input = random.uniform(shape=[2, 4, 6])
        mask = input[:, :, 0] < 0.5
        encoder(input, mask)

    def test_value_error_when_invalid_kernel_inititalizer(self):
        with self.assertRaises(ValueError):
            TransformerEncoder(
                intermediate_dim=4,
                num_heads=2,
                dropout=0.5,
                kernel_initializer="Invalid",
            )

    def test_mask_propagation(self):
        encoder = TransformerEncoder(
            intermediate_dim=4,
            num_heads=2,
        )
        inputs = random.uniform(shape=[1, 4, 6])
        mask = ops.array([[True, True, False, False]])
        inputs._keras_mask = mask
        outputs = encoder(inputs)
        self.assertAllEqual(outputs._keras_mask, mask)
