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

from keras_nlp.models.gpt_neox.gpt_neox_attention import GPTNeoXAttention
from keras_nlp.utils.keras_utils import clone_initializer

from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


class GPTNeoXDecoder(keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        num_heads,
        max_sequence_length=512,
        dropout=0.0,
        activation="relu",
        layer_norm_epsilon=1e-5,
        rotary_percentage=0.25,
        rotary_max_wavelength=10000,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        use_parallel_residual=True,
        name=None,
        **kwargs,
    ):
        self._input_shape = kwargs.pop("build_input_shape", None)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.rotary_percentage = rotary_percentage
        self.rotary_max_wavelength = rotary_max_wavelength
        self.max_sequence_length = max_sequence_length
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False
        self.supports_masking = True
        self.rotary_percentage = rotary_percentage
        self.use_parallel_residual = use_parallel_residual

        if self._input_shape is not None:
            self._build(self._input_shape)

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]

        self._input_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        # Self attention layers.
        self._self_attention_layer = GPTNeoXAttention(
            num_heads=self.num_heads,
            hidden_dim=hidden_dim,
            dropout=self.dropout,
            rotary_percentage=self.rotary_percentage,
            rotary_max_wavelength=self.rotary_max_wavelength,
            max_sequence_length=self.max_sequence_length,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )

        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Feedforward layers.
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

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
    ):
        if not self._built:
            self._build(decoder_sequence.shape)

        x = decoder_sequence  # Intermediate result.

        # Compute self attention mask.
        batch_size = tf.shape(decoder_sequence)[0]
        input_length = output_length = tf.shape(decoder_sequence)[1]

        self_attention_mask = compute_causal_mask(
            batch_size, input_length, output_length, 0
        )
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if decoder_mask is not None:
            self_attention_mask = tf.minimum(decoder_mask, self_attention_mask)

        x = self._input_layernorm(x)

        # Self attention block.
        attention_output = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
        )
        attention_output = self._self_attention_dropout(attention_output)

        if self.use_parallel_residual:
            ln_out = self._self_attention_layernorm(decoder_sequence)
            mlp_output = self._feedforward_intermediate_dense(ln_out)
            mlp_output = self._feedforward_output_dense(mlp_output)
            x = mlp_output + attention_output + decoder_sequence
        else:
            attention_output = attention_output + decoder_sequence
            ln_out = self._self_attention_layernorm(x)
            mlp_output = self._feedforward_intermediate_dense(ln_out)
            mlp_output = self.activation(mlp_output)
            mlp_output = self._feedforward_output_dense(mlp_output)
            x = mlp_output + attention_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "rotary_percentage": self.rotary_percentage,
                "rotary_max_wavelength": self.rotary_max_wavelength,
                "max_sequence_length": self.max_sequence_length,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "build_input_shape": self._input_shape,
                "use_parallel_residual": self.use_parallel_residual,
            }
        )
        return config
