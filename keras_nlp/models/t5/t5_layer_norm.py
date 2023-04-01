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
from tensorflow import keras


class T5LayerNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight", shape=(input_shape[-1],), initializer="ones"
        )

    def call(self, hidden_states):
        variance = tf.math.reduce_mean(
            tf.math.square(hidden_states), axis=-1, keepdims=True
        )
        hidden_states = hidden_states * tf.math.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states
