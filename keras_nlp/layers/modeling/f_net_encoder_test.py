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

from keras_nlp.backend import random
from keras_nlp.layers.modeling.f_net_encoder import FNetEncoder
from keras_nlp.tests.test_case import TestCase


class FNetEncoderTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=FNetEncoder,
            init_kwargs={
                "intermediate_dim": 4,
                "activation": "relu",
                "layer_norm_epsilon": 1e-5,
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
                "dropout": 0.1,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
            expected_num_trainable_weights=8,
            expected_num_non_trainable_variables=1,
        )

    def test_value_error_when_invalid_kernel_initializer(self):
        with self.assertRaises(ValueError):
            FNetEncoder(
                intermediate_dim=4,
                dropout=0.5,
                kernel_initializer="Invalid",
            )
