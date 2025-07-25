import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


class GPTNeoXAttention(keras.layers.Layer):
    """GPTNeoXAttention layer.

    This is an implementation of attention layer as described in the
    paper ["GPT-NeoX-20B: An Open-Source Autoregressive Language Model"](https://arxiv.org/abs/2204.06745).
    Effectively, this layer implements Multi-Head Self Attention with a rotary
    embedding for encoding position information.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. Hidden dimension of the input, i.e., `hidden_states`.
        bucket_size: int. The size of the relative position
            buckets. Generally equal to `max_sequence_length // 2`.
        dropout: float. Dropout probability.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense layers.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense layers.
        rotary_percentage: float. The percentage by which query, key, value
            matrices are to be rotated.
        rotary_max_wavelength: int. The maximum angular wavelength of the
            sine/cosine curves, for rotary embeddings.
        max_sequence_length: int. The maximum input sequence length.
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        rotary_percentage=0.25,
        rotary_max_wavelength=10000,
        max_sequence_length=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.rotary_percentage = rotary_percentage
        self.dropout = dropout
        self.attn_head_size = hidden_dim // num_heads
        self.rotary_max_wavelength = rotary_max_wavelength
        self.rotary_dim = int(self.attn_head_size * rotary_percentage)
        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=rotary_max_wavelength,
            dtype=self.dtype_policy,
        )
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.max_sequence_length = max_sequence_length

        self._inv_norm_factor = 1.0 / math.sqrt(self.attn_head_size)

    def build(self, input_shape):
        self._qkv_dense = keras.layers.EinsumDense(
            equation="abc,cde->abde",
            output_shape=(None, self.num_heads, 3 * self.attn_head_size),
            bias_axes="de",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="query_key_value",
        )
        self._qkv_dense.build(input_shape)

        self._attn_dropout_layer = keras.layers.Dropout(
            self.dropout,
            dtype=self.dtype_policy,
            name="attention_dropout",
        )

        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        # Output.
        self._output_dense = keras.layers.EinsumDense(
            equation="abc,cd->abd",
            output_shape=(None, self.hidden_dim),
            bias_axes="d",
            **self._get_common_kwargs_for_sublayer(use_bias=True),
            dtype=self.dtype_policy,
            name="attention_output",
        )

        self._output_dense.build(input_shape)
        self.built = True

    def _get_common_kwargs_for_sublayer(self, use_bias=True):
        common_kwargs = {}

        kernel_initializer = clone_initializer(self.kernel_initializer)
        bias_initializer = clone_initializer(self.bias_initializer)

        common_kwargs["kernel_initializer"] = kernel_initializer
        if use_bias:
            common_kwargs["bias_initializer"] = bias_initializer

        return common_kwargs

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            mask_expansion_axis = -3
            for _ in range(
                len(attention_scores.shape) - len(attention_mask.shape)
            ):
                attention_mask = ops.expand_dims(
                    attention_mask, axis=mask_expansion_axis
                )
        return self._softmax(attention_scores, attention_mask)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        if fused_attention_op_available() and self.dropout == 0:
            # Use `dot_product_attention` with Flash Attention support if
            # available.
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            attention_output = ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )
            return attention_output

        attention_scores = ops.einsum("aecd,abcd->acbe", key, query)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = self._attn_dropout_layer(
            attention_scores, training=training
        )
        attention_output = ops.einsum(
            "acbe,aecd->abcd", attention_scores, value
        )

        return attention_output

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        query_key_value = self._qkv_dense(hidden_states)

        query = query_key_value[..., : self.attn_head_size]

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update = query_key_value[
                    ..., self.attn_head_size : 2 * self.attn_head_size
                ]
                value_update = query_key_value[..., 2 * self.attn_head_size :]
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key = query_key_value[
                ..., self.attn_head_size : 2 * self.attn_head_size
            ]
            value = query_key_value[..., 2 * self.attn_head_size :]

        query_rot, query_pass = (
            query[..., : self.rotary_dim],
            query[..., self.rotary_dim :],
        )
        key_rot, key_pass = (
            key[..., : self.rotary_dim],
            key[..., self.rotary_dim :],
        )

        query_rot = self.rotary_embedding_layer(query_rot)
        key_rot = self.rotary_embedding_layer(key_rot)

        query = ops.concatenate((query_rot, query_pass), axis=-1)
        key = ops.concatenate((key_rot, key_pass), axis=-1)

        attention_output = self._compute_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            training=training,
        )

        # Reshape `attention_output` to
        # `(batch_size, sequence_length, hidden_dim)`.
        attention_output = ops.reshape(
            attention_output,
            [
                ops.shape(attention_output)[0],
                ops.shape(attention_output)[1],
                self.hidden_dim,
            ],
        )

        attention_output = self._output_dense(attention_output)

        return attention_output, cache

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "rotary_percentage": self.rotary_percentage,
                "rotary_max_wavelength": self.rotary_max_wavelength,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
