# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import string

import keras
from keras import ops

_CHR_IDX = string.ascii_lowercase


def _build_proj_equation(free_dims, bound_dims, output_dims):
    """
    Builds an einsum equation for projections inside multi-head attention.
    """
    input_str = ""
    kernel_str = ""
    output_str = ""
    bias_axes = ""
    letter_offset = 0
    for i in range(free_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    letter_offset += free_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char
    equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


def _rel_shift(x, klen=-1):
    """
    Performs relative shift to form the relative attention score.
    """

    x = ops.transpose(x, [2, 3, 0, 1])
    x_size = ops.shape(x)
    x = ops.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = ops.slice(
        x, [1, 0, 0, 0], [x_size[1] - 1, x_size[0], x_size[2], x_size[3]]
    )
    x = ops.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = ops.slice(x, [0, 0, 0, 0], [x_size[0], klen, x_size[2], x_size[3]])

    x = ops.transpose(x, [2, 3, 0, 1])

    return x


class TwoStreamRelativeAttention(keras.layers.MultiHeadAttention):
    """Two-stream relative self-attention for XLNet.

    In XLNet, each token has two associated vectors at each self-attention layer,
    the content stream (h) and the query stream (g). The content stream is the
    self-attention stream as in Transformer XL and represents the context and
    content (the token itself). The query stream only has access to contextual
    information and the position, but not the content.

    This layer shares the same build signature as `keras.layers.MultiHeadAttention`
    but has different input/output projections.

    We use the notations `B`, `T`, `S`, `M`, `L`, `E`, `P`, `dim`, `num_heads`
    below, where
    `B` is the batch dimension, `T` is the target sequence length,
    `S` in the source sequence length, `M` is the length of the state or memory,
    `L` is the length of relative positional encoding, `E` is the last dimension
    of query input, `P` is the number of predictions, `dim` is the dimensionality
    of the encoder layers. and `num_heads` is the number of attention heads.

    Args:
        content_stream: `Tensor` of shape `[B, T, dim]`.
        content_attention_bias: Bias `Tensor` for content based attention of shape
            `[num_heads, dim]`.
        positional_attention_bias: Bias `Tensor` for position based attention of
            shape `[num_heads, dim]`.
        query_stream: `Tensor` of shape `[B, P, dim]`.
        target_mapping: `Tensor` of shape `[B, P, S]`.
        relative_position_encoding: Relative positional encoding `Tensor` of
            shape `[B, L, dim]`.
        segment_matrix: Optional `Tensor` representing segmentation IDs used in
            XLNet of shape `[B, S, S + M]`.
        segment_encoding: Optional `Tensor` representing the segmentation
            encoding as used in XLNet of shape `[2, num_heads, dim]`.
        segment_attention_bias: Optional trainable bias parameter added to the
            query had when calculating the segment-based attention score used
            in XLNet of shape `[num_heads, dim]`.
        state: Optional `Tensor` of shape `[B, M, E]`.
            If passed, this is also attended over as in Transformer XL.
        content_attention_mask: a boolean mask of shape `[B, T, S]` that
            prevents attention to certain positions for content attention
            computation.
        query_attention_mask: a boolean mask of shape `[B, T, S]` that
            prevents attention to certain position for query attention
            computation.
    """

    def __init__(self, kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(kernel_initializer=kernel_initializer, **kwargs)

    def _get_common_kwargs_for_sublayer(self):
        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        return common_kwargs

    def build(self, content_stream_shape):
        self._use_bias = False

        self._query_shape = content_stream_shape
        self._key_shape = content_stream_shape
        self._value_shape = content_stream_shape

        free_dims = len(self._query_shape) - 1
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims, bound_dims=1, output_dims=2
        )
        self._query_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            dtype=self.dtype_policy,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(self._query_shape)

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            len(self._key_shape) - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            dtype=self.dtype_policy,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(self._key_shape)

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            len(self._value_shape) - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            dtype=self.dtype_policy,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(self._value_shape)

        free_dims = len(self._query_shape) - 1
        _, _, output_rank = _build_proj_equation(
            free_dims, bound_dims=2, output_dims=1
        )
        self._output_dense = keras.layers.EinsumDense(
            "ibnd,hnd->ibh",
            output_shape=_get_output_shape(
                output_rank - 1, [self._query_shape[-1]]
            ),
            bias_axes=None,
            dtype=self.dtype_policy,
            name="attention_output",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._output_dense.build(
            self._value_dense.compute_output_shape(self._value_dim)
        )

        einsum_equation, _, output_rank = _build_proj_equation(
            len(self._key_shape) - 1, bound_dims=1, output_dims=2
        )
        self._encoding_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=None,
            dtype=self.dtype_policy,
            name="encoding",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._encoding_dense.build(self._key_shape)

        self._build_attention(output_rank)
        self.built = True

    def compute_attention(
        self,
        query,
        key,
        value,
        position,
        content_attention_bias,
        positional_attention_bias,
        segment_matrix=None,
        segment_encoding=None,
        segment_attention_bias=None,
        attention_mask=None,
    ):
        """Computes the attention.

        This function defines the computation inside `call` with projected
        multihead Q, K, V, R inputs.

        We use the notations `B`, `T`, `S`, `M`, `L`, `num_heads`, `key_dim`
        below, where
        `B` is the batch dimension, `T` is the target sequence length,
        `S` in the source sequence length, `M` is the length of the state,
        `L` is the length of relative positional encoding, `num_heads` is
        number of attention heads and `key_dim` is size of each attention head
        for query and key.

        Args:
            query: Projected query `Tensor` of shape
                `[B, T, num_heads, key_dim]`.
            key: Projected key `Tensor` of shape
                `[B, S + M, num_heads, key_dim]`.
            value: Projected value `Tensor` of shape
                `[B, S + M, num_heads, key_dim]`.
            position: Projected position `Tensor` of shape
                `[B, L, num_heads, key_dim]`.
            content_attention_bias: Trainable bias parameter added to the query
                head when calculating the content-based attention score.
            positional_attention_bias: Trainable bias parameter added to the
                query head when calculating the position-based attention score.
            segment_matrix: Optional `Tensor` representing segmentation IDs
                used in XLNet.
            segment_encoding: Optional trainable `Tensor` representing the
                segmentation encoding as used in XLNet.
            segment_attention_bias: Optional trainable bias parameter added
                to the query had when calculating the segment-based attention
                score used in XLNet.
            attention_mask: (default None) Optional mask that is added to
                attention logits. If state is not None, the mask source sequence
                dimension should extend M.
        Returns:
            attention_output: Multi-headed output of attention computation of
                shape `[B, S, num_heads, key_dim]`.
        """
        content_attention = ops.einsum(
            self._dot_product_equation, key, query + content_attention_bias
        )
        positional_attention = ops.einsum(
            self._dot_product_equation,
            position,
            query + positional_attention_bias,
        )
        positional_attention = _rel_shift(
            positional_attention, klen=ops.shape(content_attention)[3]
        )

        if segment_matrix is not None:
            segment_attention = ops.einsum(
                "bind,snd->bnis",
                query + segment_attention_bias,
                segment_encoding,
            )
            target_shape = ops.shape(positional_attention)
            segment_attention = ops.where(
                ops.broadcast_to(
                    ops.expand_dims(segment_matrix, 1), target_shape
                ),
                ops.broadcast_to(segment_attention[:, :, :, 1:], target_shape),
                ops.broadcast_to(segment_attention[:, :, :, :1], target_shape),
            )
            attention_sum = (
                content_attention + positional_attention + segment_attention
            )
        else:
            attention_sum = content_attention + positional_attention

        attention_scores = ops.multiply(
            attention_sum, 1.0 / math.sqrt(float(self._key_dim))
        )

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        attention_output = self._dropout_layer(attention_scores)

        attention_output = ops.einsum(
            self._combine_equation, attention_output, value
        )

        return attention_output

    def call(
        self,
        content_stream,
        content_attention_bias,
        positional_attention_bias,
        relative_position_encoding,
        query_stream=None,
        target_mapping=None,
        segment_matrix=None,
        segment_encoding=None,
        segment_attention_bias=None,
        state=None,
        content_attention_mask=None,
        query_attention_mask=None,
    ):
        """Compute multi-head relative attention over inputs.

        We use the notations `B`, `T`, `M`, `E` below, where
        `B` is the batch dimension, `T` is the target sequence length,
        `M` is the length of the state or memory and `E` is the last
        dimension of query input.

        Args:
            content_stream: The content representation, commonly referred to as h.
                This serves a similar role to the standard hidden states in
                Transformer-XL.
            content_attention_bias: A trainable bias parameter added to the query
                head when calculating the content-based attention score.
            positional_attention_bias: A trainable bias parameter added to the
                query head when calculating the position-based attention score.
            query_stream: The query representation, commonly referred to as g.
                This only has access to contextual information and position, but
                not content. If not provided, then this is
                MultiHeadRelativeAttention with self-attention.
            relative_position_encoding: relative positional encoding for key
                and value.
            target_mapping: Optional `Tensor` representing the target mapping
                used in partial prediction.
            segment_matrix: Optional `Tensor` representing segmentation IDs
                used in XLNet.
            segment_encoding: Optional `Tensor` representing the segmentation
                encoding as used in XLNet.
            segment_attention_bias: Optional trainable bias parameter added
                to the query head when calculating the segment-based attention
                score.
            state: (default None) optional state. If passed, this is also
                attended over as in TransformerXL and XLNet.
            content_attention_mask: (default None) Optional mask that is added
                to content attention logits. If state is not None, the mask
                source sequence dimension should extend M.
            query_attention_mask: (default None) Optional mask that is added to
                query attention logits. If state is not None, the mask source
                sequence dimension should extend M.

        Returns:
            content_attention_output, query_attention_output: the results of the
                computation, both of shape `[B, T, E]`.
        """

        if state is not None and len(state.shape) > 1:
            content_and_memory_stream = ops.concatenate(
                [state, content_stream], 1
            )
        else:
            content_and_memory_stream = content_stream

        # `query` = [B, T, N, H]
        query = self._query_dense(content_stream)

        # `key` = [B, S + M, N, H]
        key = self._key_dense(content_and_memory_stream)

        # `value` = [B, S + M, N, H]
        value = self._value_dense(content_and_memory_stream)

        # `position` = [B, L, N, H]
        position = self._encoding_dense(relative_position_encoding)

        content_attention_output = self.compute_attention(
            query=query,
            key=key,
            value=value,
            position=position,
            content_attention_bias=content_attention_bias,
            positional_attention_bias=positional_attention_bias,
            segment_matrix=segment_matrix,
            segment_encoding=segment_encoding,
            segment_attention_bias=segment_attention_bias,
            attention_mask=content_attention_mask,
        )

        # `content_attention_output` = [B, S, N, H]
        content_attention_output = self._output_dense(content_attention_output)

        query_attention_output = None
        if query_stream is not None:
            query = self._query_dense(query_stream)
            if target_mapping is not None:
                query = ops.einsum("bmnd,bml->blnd", query, target_mapping)
                query_attention_output = self.compute_attention(
                    query=query,
                    key=key,
                    value=value,
                    position=position,
                    content_attention_bias=content_attention_bias,
                    positional_attention_bias=positional_attention_bias,
                    segment_matrix=segment_matrix,
                    segment_encoding=segment_encoding,
                    segment_attention_bias=segment_attention_bias,
                    attention_mask=query_attention_mask,
                )
                query_attention_output = ops.einsum(
                    "blnd,bml->bmnd", query_attention_output, target_mapping
                )
            else:
                query_attention_output = self.compute_attention(
                    query=query,
                    key=key,
                    value=value,
                    position=position,
                    content_attention_bias=content_attention_bias,
                    positional_attention_bias=positional_attention_bias,
                    segment_matrix=segment_matrix,
                    segment_encoding=segment_encoding,
                    segment_attention_bias=segment_attention_bias,
                    attention_mask=query_attention_mask,
                )
            query_attention_output = self._output_dense(query_attention_output)

        return content_attention_output, query_attention_output
