# Copyright 2024 The KerasHub Authors
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
"""Whisper decoder block."""


import keras

from keras_hub.src.layers.modeling.transformer_decoder import TransformerDecoder
from keras_hub.src.models.whisper.whisper_cached_multi_head_attention import (
    WhisperCachedMultiHeadAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer


@keras.saving.register_keras_serializable(package="keras_hub")
class WhisperDecoder(TransformerDecoder):
    """Whisper decoder.

    Inherits from `keras_hub.layers.TransformerDecoder`, and overrides the
    `build` method to use the
    `keras_hub.models.whisper.whisper_multi_head_attention.WhisperMultiHeadAttention`
    layer instead of `keras.layers.MultiHeadAttention` and
    `keras_hub.models.whisper.whisper_cached_multi_head_attention.WhisperCachedMultiHeadAttention`
    instead of `keras_hub.layers.cached_multi_head_attention.CachedMultiHeadAttention`.
    """

    def build(
        self,
        decoder_sequence_shape,
        encoder_sequence_shape,
    ):
        self._decoder_sequence_shape = decoder_sequence_shape
        self._encoder_sequence_shape = encoder_sequence_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = decoder_sequence_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        head_dim = int(hidden_dim // self.num_heads)
        if head_dim == 0:
            raise ValueError(
                "Attention `head_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        # Self attention layers.
        self._self_attention_layer = WhisperCachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention",
        )

        self._self_attention_layer.build(
            query_shape=decoder_sequence_shape,
            value_shape=decoder_sequence_shape,
        )
        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(decoder_sequence_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        self._cross_attention_layer = WhisperCachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            value_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="cross_attention",
        )
        self._cross_attention_layer.build(
            query_shape=decoder_sequence_shape,
            value_shape=encoder_sequence_shape,
        )
        self._cross_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="cross_attention_layer_norm",
        )
        self._cross_attention_layer_norm.build(decoder_sequence_shape)
        self._cross_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="cross_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(decoder_sequence_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(decoder_sequence_shape)
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        # Create layers based on input shape.
        self.built = True
