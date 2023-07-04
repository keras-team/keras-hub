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

"""Falcon Decoder Layer"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Dropout,LayerNormalization
from tensorflow.keras.activations import gelu
from keras_nlp.models.falcon.falcon_attention import FalconAttention

from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.FalconDecoderLayer")
class FalconDecoderLayer(tf.keras.Model):
    """FalconDecoder used in Falcon models for decoding.

    This layer implements the decoder layer used in Falcon models. It consists of
    self-attention mechanism followed by a feed-forward neural network (MLP).

    Args:
        config: Configuration object containing the hyperparameters.

    Inputs:

    Outputs:
  
    Examples:
        ```python
     
        ```
    """

    def __init__(self,
        num_heads,
        hidden_dim,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        max_sequence_length=512,
        name=None,
        **kwargs,
                ):
        self._input_shape = kwargs.pop("build_input_shape", None)

        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.hidden_dim=hidden_dim
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.layer_norm_epsilon = layer_norm_epsilon
        self._built = False
        if self._input_shape is not None:
            self._build(self._input_shape)
        self.input_layernorm = LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.self_attention = FalconAttention(self.num_heads,
                                              self.hidden_dim,
                                              self.dropout,
                                              max_sequence_length=512,)

        self.dense_h_to_4h = Dense(4 * hidden_dim, use_bias=False)
        self.act = gelu
        self.dense_4h_to_h = Dense(hidden_dim, use_bias=False)
        self.hidden_dropout = Dropout(self.dropout)

    def call(self, hidden_states):
        layernorm_output = self.input_layernorm(hidden_states)
        residual = hidden_states
        attn_outputs = self.self_attention(hidden_states, attention_mask=None)  # Pass attention_mask argument
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        mlp_output = self.act(self.dense_h_to_4h(self.hidden_dim))
        mlp_output = self.dense_4h_to_h(mlp_output)
        output = self.hidden_dropout(mlp_output)
        output = tf.keras.layers.Add()([output, residual])
        outputs = (output,) + outputs[1:]
        return outputs


    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]

        # Self attention layers.
        self._self_attention_layer = FalconAttention(
                                            self.num_heads,
                                            self.hidden_dim,
                                            self.dropout,
                                            max_sequence_length=512,

        )

        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        self.dense_h_to_4h = Dense(4 * hidden_dim, use_bias=False)
        self.act = gelu
        self.dense_4h_to_h = Dense(hidden_dim, use_bias=False)
        self.hidden_dropout = Dropout(self.dropout)

    def get_config(self):
        config = super().get_config()
        config.update(
            {

                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config