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
from keras_nlp.backend import keras
from keras_nlp.backend import ops


# TODO: Deprecate this in favor of
# `keras.layers.LayerNormalization(rms_scaling=True)` once Keras 2 support is
# removed.
class MistralLayerNormalization(keras.layers.Layer):
    """A normalization layer for Mistral that implements RMS normalization."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self._epsilon = epsilon

    def build(self, input_shape):
        self._dim = input_shape[-1]
        self._weight = self.add_weight(
            name="weight",
            trainable=True,
            shape=(self._dim,),
            initializer="ones",
        )
        self.built = True

    def call(self, x):
        x = x * ops.rsqrt(
            ops.mean(ops.power(x, 2), axis=-1, keepdims=True) + self._epsilon
        )
        return x * self._weight

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self._epsilon})
        return config
