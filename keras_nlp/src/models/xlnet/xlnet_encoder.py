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

import keras
from keras import ops

from keras_nlp.src.models.xlnet.relative_attention import (
    TwoStreamRelativeAttention,
)


def xlnet_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


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
     - [XLNet: Generalized Autoregressive Pretraining for Language Understanding]
     (https://arxiv.org/abs/1906.08237)
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
            dtype=self.dtype_policy,
            name="rel_attn",
        )

        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm_rel_attn",
        )
        self.layer_norm.build(input_shape)

        self.dropout_attn = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
        )

        # Feed-Forward Part
        self.layer_norm_ff = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm_ff",
        )
        self.layer_norm_ff.build(input_shape)

        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_intermediate_dense.build(input_shape)

        self.feedforward_output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        self.feedforward_output_dense.build(
            self.feedforward_intermediate_dense.compute_output_shape(
                input_shape
            )
        )

        self.dropout_ff = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
        )

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
        output_content,
        attn_mask_content,
        attn_mask_query,
        pos_emb,
        seg_mat,
        output_query=None,
        mems=None,
        target_mapping=None,
    ):
        # rel_attn
        attn_out_content, attn_out_query = self.relative_attention(
            content_stream=output_content,
            query_stream=output_query,
            content_attention_mask=attn_mask_content,
            query_attention_mask=attn_mask_query,
            relative_position_encoding=pos_emb,
            content_attention_bias=self.content_attention_bias,
            positional_attention_bias=self.positional_attention_bias,
            segment_attention_bias=self.segment_attention_bias,
            segment_matrix=seg_mat,
            segment_encoding=self.segment_encoding,
            target_mapping=target_mapping,
            state=mems,
        )

        attn_out_content = self.dropout_attn(attn_out_content)
        attn_out_content = attn_out_content + output_content
        attn_out_content = self.layer_norm(attn_out_content)

        if attn_out_query is not None:
            attn_out_query = self.dropout_attn(attn_out_query)
            attn_out_query = attn_out_query + output_query
            attn_out_query = self.layer_norm(attn_out_query)

        # feed-forward
        ff_out_content = attn_out_content
        ff_out_content = self.feedforward_intermediate_dense(ff_out_content)
        ff_out_content = self.activation_function_ff(ff_out_content)
        ff_out_content = self.dropout_ff(ff_out_content)
        ff_out_content = self.feedforward_output_dense(ff_out_content)
        ff_out_content = self.dropout_ff(ff_out_content)
        ff_out_content = self.layer_norm_ff(ff_out_content + attn_out_content)

        if attn_out_query is not None:
            ff_out_query = attn_out_query
            ff_out_query = self.feedforward_intermediate_dense(ff_out_query)
            ff_out_query = self.activation_function_ff(ff_out_query)
            ff_out_query = self.dropout_ff(ff_out_query)
            ff_out_query = self.feedforward_output_dense(ff_out_query)
            ff_out_query = self.dropout_ff(ff_out_query)
            ff_out_query = self.layer_norm_ff(ff_out_query + attn_out_query)

            return ff_out_content, ff_out_query

        return ff_out_content, None

    def compute_output_shape(
        self,
        output_content_shape,
        pos_emb_shape,
        attn_mask_content_shape,
        attn_mask_query_shape,
        seg_mat_shape,
        output_query_shape=None,
    ):
        return [output_content_shape, output_content_shape]


class XLNetAttentionMaskLayer(keras.layers.Layer):
    """
    Attention Mask Layer for XLNet Encoder Block.

    This layer processes attention masks for both content state and query state
     during the forward pass.

    Args:
        hidden_dim: int, the size hidden states.
        kernel_initializer_range: int, defaults to 0.02. The kernel initializer
            range for the dense and relative attention layers.
        **kwargs: other keyword arguments.
    """

    def __init__(self, hidden_dim, kernel_initializer_range, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.kernel_initializer_range = kernel_initializer_range
        self.kernel_initializer = xlnet_kernel_initializer(
            self.kernel_initializer_range
        )

    def build(self, inputs_shape):
        self.mask_emb = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="mask_emb",
        )
        self.built = True

    def call(self, inputs, mlen=None):
        bsz, qlen = ops.shape(inputs)[0], ops.shape(inputs)[1]
        mlen = 0 if mlen is None else mlen

        inputs = 1 - inputs
        inputs = ops.reshape(
            inputs,
            [ops.shape(inputs)[1], ops.shape(inputs)[0]],
        )

        data_mask = ops.expand_dims(inputs, 0)

        if mlen > 0:
            mems_mask = ops.zeros([ops.shape(data_mask)[0], mlen, bsz])
            data_mask = ops.concatenate(
                [ops.cast(mems_mask, dtype="int32"), data_mask], axis=1
            )
        attn_mask_query = ops.expand_dims(data_mask, -1)

        attn_mask_query = ops.cast(
            attn_mask_query > 0, dtype=attn_mask_query.dtype
        )

        # Since ops.eye doesn't support tensorflow Tensor as input.
        # we need to create custom function here.
        n = ops.expand_dims(ops.arange(qlen), -1)
        m = ops.arange(qlen)
        attn_mask_content = -ops.cast(
            ops.where(n == m, 1, 0), attn_mask_query.dtype
        )

        if mlen > 0:
            attn_mask_content = ops.concatenate(
                [
                    ops.zeros([qlen, mlen], dtype=attn_mask_content.dtype),
                    attn_mask_content,
                ],
                axis=-1,
            )

        attn_mask_content = ops.cast(
            (
                attn_mask_query
                + ops.expand_dims(ops.expand_dims(attn_mask_content, -1), -1)
            )
            > 0,
            dtype=attn_mask_content.dtype,
        )

        # to make sure inputs suitable for TwoStreamRelativeAttention
        attn_mask_content = 1.0 - ops.cast(
            ops.transpose(ops.squeeze(attn_mask_content, -1), [2, 0, 1]),
            "float32",
        )
        attn_mask_query = 1.0 - ops.cast(
            ops.transpose(ops.squeeze(attn_mask_query, -1), [2, 0, 1]),
            "float32",
        )

        return attn_mask_content, attn_mask_query

    def compute_output_shape(self, padding_mask_shape):
        return [padding_mask_shape, padding_mask_shape]


class XLNetSegmentMatrixLayer(keras.layers.Layer):
    """
    This layer creates Segment Matrix for XLNet Encoder.
    """

    def call(self, segment_ids, mlen=None):
        bsz = ops.shape(segment_ids)[0]
        mlen = 0 if mlen is None else mlen

        # Prepare seg_mat
        segment_ids = ops.transpose(segment_ids, [1, 0])

        if mlen > 0:
            mem_pad = ops.zeros([mlen, bsz], dtype=segment_ids.dtype)
            cat_ids = ops.concatenate([mem_pad, segment_ids], 0)
        else:
            cat_ids = segment_ids

        # `1` indicates not in the same segment [qlen x klen x bsz]
        seg_mat = ops.cast(
            ops.logical_not(ops.equal(segment_ids[:, None], cat_ids[None, :])),
            dtype=segment_ids.dtype,
        )

        # to make sure inputs suitable for TwoStreamRelativeAttention
        seg_mat = ops.cast(ops.transpose(seg_mat, [2, 0, 1]), dtype="bool")

        return seg_mat

    def compute_output_shape(self, segment_ids_shape):
        return segment_ids_shape
