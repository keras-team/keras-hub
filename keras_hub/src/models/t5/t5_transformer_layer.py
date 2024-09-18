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

import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.models.t5.t5_layer_norm import T5LayerNorm
from keras_hub.src.models.t5.t5_multi_head_attention import T5MultiHeadAttention


class T5TransformerLayer(keras.layers.Layer):
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        intermediate_dim,
        key_value_dim,
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
            is_decoder=is_decoder,
            hidden_dim=hidden_dim,
            key_value_dim=key_value_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_relative_attention_bias=use_relative_attention_bias,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.self_attention_layer_norm = T5LayerNorm(
            layer_norm_epsilon,
            dtype=self.dtype_policy,
        )
        self.self_attention_dropout = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

        if self.is_decoder:
            self.cross_attention = T5MultiHeadAttention(
                is_decoder=is_decoder,
                hidden_dim=hidden_dim,
                key_value_dim=key_value_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_relative_attention_bias=False,
                dtype=self.dtype_policy,
                name="cross_attention",
            )
            self.cross_attention_layer_norm = T5LayerNorm(
                layer_norm_epsilon,
                dtype=self.dtype_policy,
            )
            self.cross_attention_dropout = keras.layers.Dropout(
                dropout,
                dtype=self.dtype_policy,
            )

        self.input_projector = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            activation=keras.activations.get(activation),
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=hidden_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="input_projector",
        )
        if self.use_gated_activation:
            self.gate_projector = keras.layers.Dense(
                intermediate_dim,
                use_bias=False,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=hidden_dim**-0.5
                ),
                dtype=self.dtype_policy,
                name="gate_projector",
            )
        self.output_projector = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=intermediate_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="output_projector",
        )
        self.layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_causal_mask=False,
        training=False,
    ):
        if use_causal_mask:
            shape = ops.shape(hidden_states)
            batch_size, length = shape[0], shape[1]
            causal_mask = compute_causal_mask(batch_size, length, length)
            attention_mask = causal_mask & ops.cast(attention_mask, "bool")

        x = hidden_states  # Intermediate result.

        residual = x
        x = self.self_attention_layer_norm(x)
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
            x = self.cross_attention_layer_norm(x)
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
            hidden_linear = self.gate_projector(x)
            x = hidden_activation * hidden_linear
        else:
            x = self.input_projector(x)
        x = self.dropout_layer(x, training=training)
        x = self.output_projector(x)
        x = self.dropout_layer(x, training=training)
        x = x + residual

        if position_bias is not None:
            return x, position_bias
        else:
            return x
