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


"""Transformer encoder block implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras

import keras_nlp
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.utils.keras_utils import clone_initializer

from tensorflow import keras
from keras_nlp.layers import TwoStreamRelativeAttention

from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    merge_padding_and_attention_mask,
)

#
# @keras_nlp_export("keras_nlp.layers.XLNetEncoder")
# class XLNetEncoder(keras.layers.Layer):
#     """XLNet encoder.
#
#     This class follows the architecture of the transformer encoder layer in the
#     paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
#     can instantiate multiple instances of this class to stack up an encoder.
#
#     This layer will correctly compute an attention mask from an implicit
#     Keras padding mask (for example, by passing `mask_zero=True` to a
#     `keras.layers.Embedding` layer). See the Masking and Padding
#     [guide](https://keras.io/guides/understanding_masking_and_padding/)
#     for more details.
#
#     Args:
#         intermediate_dim: int, the hidden size of feedforward network.
#         num_heads: int, the number of heads in the
#             `keras.layers.MultiHeadRelativeAttention` layer.
#         dropout: float, defaults to 0. the dropout value, shared by
#             `keras.layers.MultiHeadRelativeAttention` and feedforward network.
#         activation: string or `keras.activations`, defaults to "relu". the
#             activation function of feedforward network.
#         layer_norm_epsilon: float, defaults to 1e-5. The epsilon value in layer
#             normalization components.
#         kernel_initializer: string or `keras.initializers` initializer,
#             defaults to "glorot_uniform". The kernel initializer for
#             the dense and multiheaded relative attention layers.
#         bias_initializer: string or `keras.initializers` initializer,
#             defaults to "zeros". The bias initializer for
#             the dense and multiheaded relative attention layers.
#         normalize_first: bool. Defaults to False. If True, the inputs to the
#             attention layer and the intermediate dense layer  are normalized
#             (similar to GPT-2). If set to False, outputs of attention layer and
#             intermediate dense layer are normalized (similar to XLNet).
#         name: string, defaults to None. The name of the layer.
#         **kwargs: other keyword arguments.
#
#     Examples:
#
#     ```python
#     # Create a single transformer encoder layer.
#     encoder = keras_nlp.layers.XLNetEncoder(
#         intermediate_dim=64, num_heads=8)
#
#     # Create a simple model containing the encoder.
#     input = keras.Input(shape=[10, 64])
#     output = encoder(input)
#     model = keras.Model(inputs=input, outputs=output)
#
#     # Call encoder on the inputs.
#     input_data = tf.random.uniform(shape=[2, 10, 64])
#     output = model(input_data)
#     ```
#
#     References:
#      - [](https://arxiv.org/abs/1906.08237)
#     """
#
#     def __init__(
#         self,
#         intermediate_dim,
#         num_heads,
#         dropout=0,
#         activation="relu",
#         layer_norm_epsilon=1e-05,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         normalize_first=False,
#         name=None,
#         **kwargs,
#     ):
#         # Work around for model saving
#         self._input_shape = kwargs.pop("build_input_shape", None)
#
#         super().__init__(name=name, **kwargs)
#         self.intermediate_dim = intermediate_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.activation = keras.activations.get(activation)
#         self.layer_norm_epsilon = layer_norm_epsilon
#         self.kernel_initializer = keras.initializers.get(kernel_initializer)
#         self.bias_initializer = keras.initializers.get(bias_initializer)
#         self.normalize_first = normalize_first
#         self._built = False
#         self.supports_masking = True
#
#         if self._input_shape is not None:
#             self._build(self._input_shape)
#
#     def _build(self, input_shape):
#         # Create layers based on input shape.
#         self._built = True
#         self._input_shape = input_shape
#         # Infer the dimension of our hidden feature size from the build shape.
#         hidden_dim = input_shape[-1]
#         # Attention head size is `hidden_dim` over the number of heads.
#         key_dim = int(hidden_dim // self.num_heads)
#
#         # Relaive attention layers.
#         self._relative_attention_layer = (
#             keras_nlp.layers.MultiHeadRelativeAttention(
#                 num_heads=self.num_heads,
#                 key_dim=key_dim,
#                 dropout=self.dropout,
#                 kernel_initializer=clone_initializer(self.kernel_initializer),
#                 bias_initializer=clone_initializer(self.bias_initializer),
#             )
#         )
#         self._relative_attention_layer._build_from_signature(
#             query=input_shape,
#             value=input_shape,
#             content_attention_bias=self.bias_param1,
#             positional_attention_bias=self.bias_param2,
#         )
#
#         # Feedforward layers.
#         self._feedforward_layernorm = keras.layers.LayerNormalization(
#             epsilon=self.layer_norm_epsilon,
#         )
#         self._feedforward_intermediate_dense = keras.layers.Dense(
#             self.intermediate_dim,
#             activation=self.activation,
#             kernel_initializer=clone_initializer(self.kernel_initializer),
#             bias_initializer=clone_initializer(self.bias_initializer),
#         )
#         self._feedforward_output_dense = keras.layers.Dense(
#             hidden_dim,
#             kernel_initializer=clone_initializer(self.kernel_initializer),
#             bias_initializer=clone_initializer(self.bias_initializer),
#         )
#         self._feedforward_dropout = keras.layers.Dropout(
#             rate=self.dropout,
#         )
#
#         self.bias_param1 = tf.Variable(
#             shape=input_shape,
#             name="bias_param1",
#             initializer=tf.zeros_initializer(),
#         )
#         self.bias_param2 = tf.Variable(
#             shape=input_shape,
#             name="bias_param2",
#             initializer=tf.zeros_initializer(),
#         )
#
#     def call(self, inputs, padding_mask=None, attention_mask=None):
#         """Forward pass of the XLNetEncoder.
#
#         Args:
#             inputs: a Tensor. The input data to XLNetEncoder, should be
#                 of shape [batch_size, sequence_length, hidden_dim].
#             padding_mask: a boolean Tensor. It indicates if the token should be
#                 masked because the token is introduced due to padding.
#                 `padding_mask` should have shape [batch_size, sequence_length].
#             attention_mask: a boolean Tensor. Customized mask used to mask out
#                 certain tokens. `attention_mask` should have shape
#                 [batch_size, sequence_length, sequence_length].
#
#         Returns:
#             A Tensor of the same shape as the `inputs`.
#         """
#
#         if not self._built:
#             self._build(inputs.shape)
#
#         x = inputs  # Intermediate result.
#
#         # Compute self attention mask.
#         relative_attention_mask = merge_padding_and_attention_mask(
#             inputs, padding_mask, attention_mask
#         )
#
#         # Self attention block.
#         residual = x
#         if self.normalize_first:
#             x = self._relative_attention_layernorm(x)
#         x = self._relative_attention_layer(
#             query=x,
#             value=x,
#             attention_mask=relative_attention_mask,
#             content_attention_bias=self.bias_param1,
#             positional_attention_bias=self.bias_param2,
#         )
#         x = self._relative_attention_dropout(x)
#         x = x + residual
#         if not self.normalize_first:
#             x = self._relative_attention_layernorm(x)
#
#         # Feedforward block.
#         residual = x
#         if self.normalize_first:
#             x = self._feedforward_layernorm(x)
#         x = self._feedforward_intermediate_dense(x)
#         x = self._feedforward_output_dense(x)
#         x = self._feedforward_dropout(x)
#         x = x + residual
#         if not self.normalize_first:
#             x = self._feedforward_layernorm(x)
#
#         return x
#
#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "intermediate_dim": self.intermediate_dim,
#                 "num_heads": self.num_heads,
#                 "dropout": self.dropout,
#                 "activation": keras.activations.serialize(self.activation),
#                 "layer_norm_epsilon": self.layer_norm_epsilon,
#                 "kernel_initializer": keras.initializers.serialize(
#                     self.kernel_initializer
#                 ),
#                 "bias_initializer": keras.initializers.serialize(
#                     self.bias_initializer
#                 ),
#                 "normalize_first": self.normalize_first,
#                 "build_input_shape": self._input_shape,
#             }
#         )
#         return config
#
#



from tensorflow import keras
from keras_nlp.layers import TwoStreamRelativeAttention

class XLNetEncoder(keras.layers.Layer):
    def __init__(self,
                 intermediate_dim,
                 num_heads,
                 head_dim,
                 ff_dim,
                 dropout=0,
                 layer_norm_epsilon=1e-12,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 name=None,
                 **kwargs
    ):

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False
        self.supports_masking = True

        # Attention Part
        self.relative_attention = TwoStreamRelativeAttention(num_heads=self.num_heads,
                                                key_dim=self.head_dim,
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer=self.bias_initializer,
                                                name="rel_attn",
                                                )
        self.layer_norm = keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon,
                                                             name="layer_norm_rel_attn")
        self.dropout = keras.layers.Dropout(self.dropout)

        # Feed-Forward Part
        self.layer_norm_ff = keras.layers.LayerNormalization(epsilon=self.layer_norm_epsilon,
                                                             name="layer_norm_ff")
        self.layer_1_ff = tf.keras.layers.Dense(
            self.ff_dim, kernel_initializer=self.kernel_initializer, name="layer_1_ff"
        )
        self.layer_2_ff = tf.keras.layers.Dense(
            self.intermediate_dim, kernel_initializer=self.kernel_initializer, name="layer_2_ff"
        )
        self.dropout_ff = tf.keras.layers.Dropout(config.dropout)
        self.activation_function_ff = keras.activations.gelu

    def build(self, input_shape):
        self.content_attention_bias = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer=self.bias_initializer,
            trainable=True,
            name="content_attention_bias"
        )
        self.positional_attention_bias = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer=self.bias_initializer,
            trainable=True,
            name="positional_attention_bias"
        )
        self.segment_attention_bias = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer=self.bias_initializer,
            trainable=True,
            name="segment_attention_bias"
        )
        self.segment_encoding = self.add_weight(
            shape=(2, self.num_heads, self.head_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="segment_encoding"
        )
        super().build(input_shape)

    def call(self,
             output_h,
             output_g,
             attn_mask_h,
             attn_mask_g,
             pos_emb,
             seg_mat=None,
             mems=None,
             target_mapping=None,
             training=False,
        ):

        # rel_attn
        attn_out_h, attn_out_g = self.relative_attention(content_stream=output_h,
                                           query_stream=output_g,
                                           content_attention_mask=attn_mask_h,
                                           query_attention_mask=attn_mask_g,
                                           relative_position_encoding=pos_emb,
                                           content_attention_bias=self.content_attention_bias,
                                           positional_attention_bias=self.positional_attention_bias,
                                           segment_attention_bias=self.segment_attention_bias,
                                           segment_matrix=seg_mat,
                                           segment_encoding=self.segment_encoding,
                                           target_mapping=target_mapping,
                                           state=mems,
                                           )
        attn_out_h = self.dropout(attn_out_h, training=training)
        attn_out_h = attn_out_h + output_h
        attn_out_h = self.layer_norm(attn_out_h)

        if attn_out_g is not None:
            attn_out_g = self.dropout(attn_out_g)
            attn_out_g = attn_out_g + output_g
            attn_out_g = self.layer_norm(attn_out_g)

        # feed-forward
        ff_out_h = attn_out_h
        ff_out_h = self.layer_1_ff(ff_out_h)
        ff_out_h = self.activation_function_ff(ff_out_h)
        ff_out_h = self.dropout_ff(ff_out_h, training=training)
        ff_out_h = self.layer_2_ff(ff_out_h)
        ff_out_h = self.dropout_ff(ff_out_h, training=training)
        ff_out_h = self.layer_norm_ff(ff_out_h + attn_out_h)

        if attn_out_g is not None:
            ff_out_g = attn_out_g
            ff_out_g = self.layer_1_ff(ff_out_g)
            ff_out_g = self.activation_function_ff(ff_out_g)
            ff_out_g = self.dropout_ff(ff_out_g, training=training)
            ff_out_g = self.layer_2_ff(ff_out_g)
            ff_out_g = self.dropout_ff(ff_out_g, training=training)
            ff_out_g = self.layer_norm_ff(ff_out_g + attn_out_g)

            return ff_out_h, ff_out_g

        return ff_out_h, None



