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
"""Whisper encoder block."""

from tensorflow import keras

from keras_nlp.layers.transformer_encoder import TransformerEncoder


@keras.utils.register_keras_serializable(package="keras_nlp")
class WhisperEncoder(TransformerEncoder):
    """A Whisper encoder.

    Inherits from `keras_nlp.layers.TransformerEncoder`, and overrides the
    `_build` method so as to remove the bias term from the key projection layer.
    """

    def _build(self, input_shape):
        super()._build(input_shape)

        # Since there is no exposed option for this in MHA, we will reach into
        # the internals of the layer for now.
        self._self_attention_layer._key_dense.bias_axes = None
