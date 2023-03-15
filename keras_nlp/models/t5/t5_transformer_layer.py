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
from tensorflow import keras

from keras_nlp.models.t5.t5_layer_norm import T5LayerNorm
from keras_nlp.models.t5.t5_multi_head_attention import T5MultiHeadAttention


class T5TransformerLayer(keras.layers.Layer):
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        intermediate_dim,
        dropout,
        activation,
        layer_norm_epsilon,
        num_heads,
        use_gated_activation=False,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_decoder = is_decoder
        self.use_gated_activation = use_gated_activation

        self.self_attention = T5MultiHeadAttention(
            is_decoder,
            hidden_dim,
            num_heads,
            dropout,
            use_relative_attention_bias=use_relative_attention_bias,
        )
        self.self_attention_layernorm = T5LayerNorm(layer_norm_epsilon)
        self.self_attention_dropout = keras.layers.Dropout(dropout)

        if self.is_decoder:
            self.cross_attention = T5MultiHeadAttention(
                is_decoder,
                hidden_dim,
                num_heads,
                dropout,
                use_relative_attention_bias=False,
            )
            self.cross_attention_layernorm = T5LayerNorm(layer_norm_epsilon)
            self.cross_attention_dropout = keras.layers.Dropout(dropout)

        self.input_projector = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            name="input_projector",
            activation=keras.activations.get(activation),
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=hidden_dim**-0.5
            ),
        )
        if self.use_gated_activation:
            self.gate_projector = keras.layers.Dense(
                intermediate_dim,
                use_bias=False,
                name="gate_projector",
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=hidden_dim**-0.5
                ),
            )
        self.output_projector = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="output_projector",
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=intermediate_dim**-0.5
            ),
        )
        self.layer_norm = T5LayerNorm(epsilon=layer_norm_epsilon)
        self.dropout_layer = keras.layers.Dropout(dropout)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        training=False,
    ):
        x = hidden_states  # Intermediate result.

        residual = x
        x = self.self_attention_layernorm(x)
        x, position_bias = self.self_attention(
            x,
            mask=attention_mask,
            position_bias=position_bias,
            training=training,
        )
        x = self.self_attention_dropout(x, training=training)
        x = x + residual

        if self.is_decoder:
            residual = x
            x = self.cross_attention_layernorm(x)
            x, _ = self.cross_attention(
                x,
                key_value_states=encoder_hidden_states,
                mask=encoder_attention_mask,
                training=training,
            )
            x = self.cross_attention_dropout(x, training=training)
            x = x + residual

        residual = x
        x = self.layer_norm(x)
        if self.use_gated_activation:
            hidden_activation = self.input_projector(x)
            hidden_linear = self.gate_projector(hidden_states)
            x = hidden_activation * hidden_linear
        else:
            x = self.input_projector(x)
        x = self.dropout_layer(x, training=training)
        x = self.output_projector(x)
        x = self.dropout_layer(x, training=training)
        x = x + residual

        return x, position_bias
