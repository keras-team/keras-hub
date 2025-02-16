import keras

from keras_hub.src.models.moonshine.moonshine_utils import _apply_rotary_pos_emb
from keras_hub.src.models.whisper.whisper_cached_multi_head_attention import (
    _build_proj_equation,
)
from keras_hub.src.models.whisper.whisper_cached_multi_head_attention import (
    _get_output_shape,
)


# Note: Not subclassed from WhisperCachedMultiHeadAttention.
# Although the functional overlap is high and short-term value differences are
# only around 1e-4, these minor discrepancies accumulate and lead to significant
# deviations during the construction of the encoder block.
@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineMultiHeadAttention(keras.layers.MultiHeadAttention):
    """A Multi-Head Attention layer with rotary positional embeddings.

    A variant of Keras's MultiHeadAttention that incorporates rotary positional
    embeddings (RoPE) directly into the attention computation. This class
    follows the architecture used in KerasHub's Whisper implementation.

    Args:
        All arguments are inherited from keras.layers.MultiHeadAttention.
        rot_pos_emb: Tensor containing rotary positional embeddings to be
        applied to queries and keys.

    The layer projects queries, keys, and values, applies rotary positional
    embeddings, computes scaled dot-product attention, and projects the output.

    Example:

    ```python
    import keras
    import numpy as np

    from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
        MoonshineMultiHeadAttention
    )

    batch_size = 2
    seq_len = 10
    embedding_dim = 64
    num_heads = 8
    query_tensor = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    value_tensor = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    key_tensor = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    positional_embedding_tensor = keras.ops.convert_to_tensor(
        np.random.randn(seq_len, embedding_dim).astype("float32")
    )
    attention = MoonshineMultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
    )
    output = attention(
        query=query_tensor,
        value=value_tensor,
        key=key_tensor,
        rot_pos_emb=positional_embedding_tensor,
    )
    print(output)
    ```
    """

    def build(self, query_shape, value_shape, key_shape=None):
        # Ensure key_shape is defined.
        key_shape = value_shape if key_shape is None else key_shape
        query_rank = len(query_shape)
        value_rank = len(value_shape)
        key_rank = len(key_shape)

        # Build query projection layer.
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims=query_rank - 1, bound_dims=1, output_dims=2
        )
        self._query_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="query",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._query_dense.build(query_shape)

        # Build key projection layer.
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims=key_rank - 1, bound_dims=1, output_dims=2
        )
        self._key_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._key_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="key",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._key_dense.build(key_shape)

        # Build value projection layer.
        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims=value_rank - 1, bound_dims=1, output_dims=2
        )
        self._value_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(
                output_rank - 1, [self._num_heads, self._value_dim]
            ),
            bias_axes=bias_axes if self._use_bias else None,
            name="value",
            **self._get_common_kwargs_for_sublayer(),
        )
        self._value_dense.build(value_shape)

        # Build the internal attention computation sublayer.
        self._build_attention(output_rank)

        # Build output projection layer.
        if self._output_shape:
            if isinstance(self._output_shape, (list, tuple)):
                output_shape = list(self._output_shape)
            else:
                output_shape = [self._output_shape]
        else:
            output_shape = [query_shape[-1]]

        einsum_equation, bias_axes, output_rank = _build_proj_equation(
            free_dims=query_rank - 1,
            bound_dims=2,
            output_dims=len(output_shape),
        )
        self._output_dense = keras.layers.EinsumDense(
            einsum_equation,
            output_shape=_get_output_shape(output_rank - 1, output_shape),
            bias_axes=bias_axes if self._use_bias else None,
            name="attention_output",
            **self._get_common_kwargs_for_sublayer(),
        )
        output_dense_input_shape = list(
            self._query_dense.compute_output_shape(query_shape)
        )
        output_dense_input_shape[-1] = self._value_dim
        self._output_dense.build(tuple(output_dense_input_shape))

        self.built = True

    def call(self, query, value, key, rot_pos_emb, training=None, **kwargs):
        # Project query, key, and value.
        query_proj = self._query_dense(query)
        key_proj = self._key_dense(key)
        value_proj = self._value_dense(value)
        # Apply rotary positional embeddings to query and key.
        query_proj = _apply_rotary_pos_emb(query_proj, rot_pos_emb)
        key_proj = _apply_rotary_pos_emb(key_proj, rot_pos_emb)
        # Compute attention.
        attention_output, attention_scores = self._compute_attention(
            query=query_proj,
            key=key_proj,
            value=value_proj,
            training=training,
            **kwargs,
        )
        # Project the attention output.
        output = self._output_dense(attention_output)
        return output

    def get_config(self):
        config = super().get_config()
        return config
