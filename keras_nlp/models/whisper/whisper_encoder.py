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
from keras_nlp.models.whisper.whisper_multi_head_attention import (
    WhisperMultiHeadAttention,
)
from keras_nlp.utils.keras_utils import clone_initializer


class WhisperEncoder(TransformerEncoder):
    """Whisper encoder.

    Inherits from `keras_nlp.layers.TransformerEncoder`, and overrides the
    `_build` method to use the
    `keras_nlp.models.whisper.whisper_multi_head_attention.WhisperMultiHeadAttention`
    layer instead of `keras.layers.MultiHeadAttention`.
    """

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)

        # Self attention layers.
        self._self_attention_layer = WhisperMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._self_attention_layer._build_from_signature(
            query=input_shape,
            value=input_shape,
        )
        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Feedforward layers.
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )
