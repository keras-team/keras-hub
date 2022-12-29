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

from keras_nlp.layers.transformer_encoder import TransformerEncoder


class AlbertGroupLayer(keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        num_heads,
        intermediate_dim,
        activation,
        dropout,
        kernel_initializer,
        name,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.activation = keras.activations.get(activation)
        self.dropout = dropout
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # Define Transformer blocks.
        self.transformer_layers = [
            TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=activation,
                dropout=dropout,
                kernel_initializer=kernel_initializer,
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]

    def call(self, inputs, padding_mask):
        x = inputs
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "activation": keras.activations.serialize(self.activation),
                "dropout": self.dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
