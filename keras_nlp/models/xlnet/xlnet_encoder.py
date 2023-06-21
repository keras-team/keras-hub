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


"""XLNet Encoder block implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers import TwoStreamRelativeAttention
from keras_nlp.models.xlnet.xlnet_content_and_query_embedding import (
    xlnet_kernel_initializer,
)


@keras_nlp_export("keras_nlp.layers.XLNetEncoder")
class XLNetEncoder(keras.layers.Layer):
    """
    XLNet Encoder.

    This class follows the architecture of the transformer encoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up an encoder.

    Contrary to the single hidden state used in the paper mentioned above, this
    Encoder uses two hidden states, Content State and Query State. Thus calculates
    Two Stream Relative Attention using both of the hidden states. To know more
    please check the reference.

    Args:
        num_heads: int, the number of heads in the
            `keras.layers.TwoStreamRelativeAttention` layer.
        hidden_dim: int, the size hidden states.
        head_dim: int, the size of each attention head.
        intermediate_dim: int, the hidden size of feedforward network.
        dropout: float, defaults to 0.0 the dropout value, shared by
            `keras.layers.TwoStreamRelativeAttention` and feedforward network.
        activation: string or `keras.activations`, defaults to "gelu". the
            activation function of feedforward network.
        layer_norm_epsilon: float, defaults to 1e-12. The epsilon value in layer
            normalization components.
        kernel_initializer_range: int, defaults to 0.02. The kernel initializer
            range for the dense and relative attention layers.
        bias_initializer: string or `keras.initializers` initializer,
            defaults to "zeros". The bias initializer for
            the dense and multiheaded relative attention layers.
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    References:
     - [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        head_dim,
        intermediate_dim,
        dropout=0.0,
        activation="gelu",
        layer_norm_epsilon=1e-12,
        kernel_initializer_range=0.02,
        bias_initializer="zeros",
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer_range = kernel_initializer_range
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_initializer = xlnet_kernel_initializer(
            self.kernel_initializer_range
        )

    def build(self, input_shape):
        # Attention Part
        self.relative_attention = TwoStreamRelativeAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="rel_attn",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="layer_norm_rel_attn"
        )
        self.dropout = keras.layers.Dropout(self.dropout)

        # Feed-Forward Part
        self.layer_norm_ff = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="layer_norm_ff"
        )
        self.layer_1_ff = tf.keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            name="layer_1_ff",
        )
        self.layer_2_ff = tf.keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            name="layer_2_ff",
        )
        self.dropout_ff = tf.keras.layers.Dropout(self.dropout)
        self.activation_function_ff = keras.activations.get(self.activation)

        self.content_attention_bias = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer=self.bias_initializer,
            trainable=True,
            name="content_attention_bias",
        )
        self.positional_attention_bias = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer=self.bias_initializer,
            trainable=True,
            name="positional_attention_bias",
        )
        self.segment_attention_bias = self.add_weight(
            shape=(self.num_heads, self.head_dim),
            initializer=self.bias_initializer,
            trainable=True,
            name="segment_attention_bias",
        )
        self.segment_encoding = self.add_weight(
            shape=(2, self.num_heads, self.head_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="segment_encoding",
        )
        super().build(input_shape)

    def call(
        self,
        output_h,
        output_g,
        attn_mask_h,
        attn_mask_g,
        pos_emb,
        seg_mat=None,
        mems=None,
        target_mapping=None,
    ):
        # rel_attn
        attn_out_h, attn_out_g = self.relative_attention(
            content_stream=output_h,
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
        attn_out_h = self.dropout(attn_out_h)
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
        ff_out_h = self.dropout_ff(ff_out_h)
        ff_out_h = self.layer_2_ff(ff_out_h)
        ff_out_h = self.dropout_ff(ff_out_h)
        ff_out_h = self.layer_norm_ff(ff_out_h + attn_out_h)

        if attn_out_g is not None:
            ff_out_g = attn_out_g
            ff_out_g = self.layer_1_ff(ff_out_g)
            ff_out_g = self.activation_function_ff(ff_out_g)
            ff_out_g = self.dropout_ff(ff_out_g)
            ff_out_g = self.layer_2_ff(ff_out_g)
            ff_out_g = self.dropout_ff(ff_out_g)
            ff_out_g = self.layer_norm_ff(ff_out_g + attn_out_g)

            return ff_out_h, ff_out_g

        return ff_out_h, None
