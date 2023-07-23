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


from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.models.xlnet.relative_attention import TwoStreamRelativeAttention


def xlnet_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


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
            name="rel_attn",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="layer_norm_rel_attn"
        )
        self.dropout_attn = keras.layers.Dropout(self.dropout)

        # Feed-Forward Part
        self.layer_norm_ff = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon, name="layer_norm_ff"
        )
        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=self.kernel_initializer,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            name="feedforward_output_dense",
        )
        self.dropout_ff = keras.layers.Dropout(self.dropout)
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
        attn_out_h = self.dropout_attn(attn_out_h)
        attn_out_h = attn_out_h + output_h
        attn_out_h = self.layer_norm(attn_out_h)

        if attn_out_g is not None:
            attn_out_g = self.dropout_attn(attn_out_g)
            attn_out_g = attn_out_g + output_g
            attn_out_g = self.layer_norm(attn_out_g)

        # feed-forward
        ff_out_h = attn_out_h
        ff_out_h = self.feedforward_intermediate_dense(ff_out_h)
        ff_out_h = self.activation_function_ff(ff_out_h)
        ff_out_h = self.dropout_ff(ff_out_h)
        ff_out_h = self.feedforward_output_dense(ff_out_h)
        ff_out_h = self.dropout_ff(ff_out_h)
        ff_out_h = self.layer_norm_ff(ff_out_h + attn_out_h)

        if attn_out_g is not None:
            ff_out_g = attn_out_g
            ff_out_g = self.feedforward_intermediate_dense(ff_out_g)
            ff_out_g = self.activation_function_ff(ff_out_g)
            ff_out_g = self.dropout_ff(ff_out_g)
            ff_out_g = self.feedforward_output_dense(ff_out_g)
            ff_out_g = self.dropout_ff(ff_out_g)
            ff_out_g = self.layer_norm_ff(ff_out_g + attn_out_g)

            return ff_out_h, ff_out_g

        return ff_out_h, None


class XLNetEncoderBlockPreprocessingLayer(keras.layers.Layer):
    """
    Preprocessing Layer for XLNet Encoder Block.

    This layer creates relative_positional_encoding and processes attention
    masks for both states during the forward pass. It binds all the complex
    logic required by the XLNet Encoder.

    In addition to that it also processes perm_mask and target_mapping tensors
    during pretraining and when mems are used.

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

        self._built = None

    def build(self, input_shape):
        self.mask_emb = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            name="mask_emb",
        )
        super().build(input_shape)

    def call(
        self,
        token_id_input,
        word_emb,
        padding_mask,
        segment_ids,
        mlen=None,
        perm_mask=None,
        target_mapping=None,
    ):
        if not self._built:
            self.build((1, 1))
            self._built = True

        bsz, qlen = ops.shape(token_id_input)[0], ops.shape(token_id_input)[1]
        mlen = 0 if mlen is None else mlen

        padding_mask = 1 - padding_mask
        padding_mask = ops.reshape(
            padding_mask,
            [ops.shape(padding_mask)[1], ops.shape(padding_mask)[0]],
        )
        perm_mask = (
            ops.transpose(perm_mask, [1, 2, 0])
            if perm_mask is not None
            else perm_mask
        )
        target_mapping = (
            ops.transpose(target_mapping, [1, 2, 0])
            if target_mapping is not None
            else target_mapping
        )

        if padding_mask is not None and perm_mask is not None:
            data_mask = padding_mask[None] + perm_mask
        elif padding_mask is not None and perm_mask is None:
            data_mask = padding_mask[None]
        elif padding_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            if mlen > 0:
                mems_mask = ops.zeros([ops.shape(data_mask)[0], mlen, bsz])
                data_mask = ops.concatenate(
                    [ops.cast(mems_mask, dtype="int32"), data_mask], axis=1
                )
            attn_mask_g = data_mask[:, :, :, None]
        else:
            attn_mask_g = None

        if attn_mask_g is not None:
            attn_mask_g = ops.cast(attn_mask_g > 0, dtype=attn_mask_g.dtype)

            # Since ops.eye doesnt support tf Tensor as input.
            attn_mask_h = ops.zeros([qlen, qlen], dtype=attn_mask_g.dtype)
            updates = ops.ones([qlen], dtype=attn_mask_g.dtype)
            indices = ops.transpose(
                ops.array(
                    [ops.arange(qlen), ops.arange(qlen)],
                    dtype=attn_mask_g.dtype,
                ),
                [1, 0],
            )
            attn_mask_h = -ops.scatter_update(attn_mask_h, indices, updates)

            if mlen > 0:
                attn_mask_h = ops.concatenate(
                    [
                        ops.zeros([qlen, mlen], dtype=attn_mask_h.dtype),
                        attn_mask_h,
                    ],
                    axis=-1,
                )

            attn_mask_h = ops.cast(
                (attn_mask_g + attn_mask_h[:, :, None, None]) > 0,
                dtype=attn_mask_h.dtype,
            )
        else:
            attn_mask_h = None

        # Prepare h & g hidden states
        output_h = word_emb
        if target_mapping is not None:
            import tensorflow as tf

            word_emb_q = tf.tile(
                self.mask_emb, [ops.shape(target_mapping)[0], bsz, 1]
            )
            output_g = self.dropout_layer(word_emb_q)
        else:
            output_g = None

        segment_ids = (
            ops.transpose(segment_ids, [1, 0])
            if segment_ids is not None
            else None
        )
        # Segment embedding
        if segment_ids is not None:
            if mlen > 0:
                mem_pad = ops.zeros([mlen, bsz], dtype=segment_ids.dtype)
                cat_ids = ops.concatenate([mem_pad, segment_ids], 0)
            else:
                cat_ids = segment_ids

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = ops.cast(
                ops.logical_not(
                    ops.equal(segment_ids[:, None], cat_ids[None, :])
                ),
                dtype=segment_ids.dtype,
            )
        else:
            seg_mat = None

        # to make sure inputs suitable for TwoStreamRelativeAttention
        output_g = (
            ops.reshape(
                output_g,
                [
                    ops.shape(output_g)[1],
                    ops.shape(output_g)[0],
                    ops.shape(output_g)[2],
                ],
            )
            if output_g is not None
            else None
        )
        attn_mask_h = (
            1.0
            - ops.cast(
                ops.transpose(ops.squeeze(attn_mask_h, -1), [2, 0, 1]),
                "float32",
            )
            if attn_mask_h is not None
            else None
        )
        attn_mask_g = (
            1.0
            - ops.cast(
                ops.transpose(ops.squeeze(attn_mask_g, -1), [2, 0, 1]),
                "float32",
            )
            if attn_mask_g is not None
            else None
        )

        seg_mat = (
            ops.cast(ops.transpose(seg_mat, [2, 0, 1]), dtype="bool")
            if seg_mat is not None
            else None
        )
        target_mapping = (
            ops.cast(
                ops.reshape(
                    target_mapping,
                    [
                        ops.shape(target_mapping)[2],
                        ops.shape(target_mapping)[0],
                        ops.shape(target_mapping)[1],
                    ],
                ),
                "float32",
            )
            if target_mapping is not None
            else None
        )

        return (
            output_h,
            output_g,
            target_mapping,
            seg_mat,
            attn_mask_h,
            attn_mask_g,
        )
