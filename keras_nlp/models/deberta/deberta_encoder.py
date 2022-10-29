# Copyright 2022 The KerasNLP Authors
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
from tensorflow import keras

from keras_nlp.models.deberta.disentangled_self_attention import (
    DisentangledSelfAttention,
)
from keras_nlp.utils.keras_utils import clone_initializer

from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    merge_padding_and_attention_mask,
)


class DebertaEncoder(keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        num_heads,
        max_position_embeddings=512,
        bucket_size=256,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs
    ):
        # Work around for model saving
        self._input_shape = kwargs.pop("build_input_shape", None)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.bucket_size = bucket_size
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False
        self.supports_masking = True

        if self._input_shape is not None:
            self._build(self._input_shape)

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]

        # Self attention layers.
        self._self_attention_layer = DisentangledSelfAttention(
            num_heads=self.num_heads,
            hidden_dim=hidden_dim,
            max_position_embeddings=self.max_position_embeddings,
            bucket_size=self.bucket_size,
            dropout=self.dropout,
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

    def call(
        self, inputs, rel_embeddings, padding_mask=None, attention_mask=None
    ):

        if not self._built:
            self._build(inputs.shape)

        x = inputs  # Intermediate result.

        # Compute self attention mask.
        self_attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, attention_mask
        )

        # Self attention block.
        residual = x
        x = self._self_attention_layer(
            hidden_states=x,
            rel_embeddings=rel_embeddings,
            attention_mask=self_attention_mask,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        x = self._self_attention_layernorm(x)

        # Feedforward block.
        residual = x
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        x = x + residual
        x = self._feedforward_layernorm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "build_input_shape": self._input_shape,
            }
        )
        return config
