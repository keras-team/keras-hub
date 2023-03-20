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
"""Tests for sampler."""

# import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.samplers import ContrastiveSampler


class ContrastiveSamplerTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.vocab_size = 10
        self.feature_size = 16

        # Create a dummy model to predict the next token.
        model = keras.Sequential(
            [
                keras.Input(shape=[None]),
                keras.layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.feature_size,
                ),
                keras.layers.Dense(self.vocab_size),
                keras.layers.Softmax(),
            ]
        )

        def token_probability_fn(inputs, mask):
            return model(inputs)

        self.token_probability_fn = token_probability_fn
        self.sampler = ContrastiveSampler(k=2)
