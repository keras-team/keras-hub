import keras

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
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


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineCausalMultiHeadAttention(CachedMultiHeadAttention):
    """A Causal Multi-Head Attention layer with rotary positional embeddings.

    A variant of Keras's CachedMultiHeadAttention that incorporates rotary
    positional embeddings (RoPE) and causal masking into the attention
    computation. This class follows the architecture used in KerasHub's Whisper
    implementation.

    Args:
        Inherits arguments from `keras_hub.layers.CachedMultiHeadAttention`.
        rot_pos_emb: Tensor containing rotary positional embeddings to be
            applied to queries and keys.
        value_cache: Optional values cache from previous attention computations.
        key_cache: Optional keys cache from previous attention computations.

    The layer projects queries, keys, and values, applies rotary positional
    embeddings, computes masked scaled dot-product attention, and projects the
    output. It also supports caching of key and value states for efficient
    autoregressive generation.

    Returns:
        A tuple of (attention_output, key_state, value_state) where
        attention_output is the processed attention output, and key_state and
        value_state are the updated cache states.

    Example:

    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
        MoonshineCausalMultiHeadAttention
    )

    batch_size = 2
    seq_len = 10
    embedding_dim = 64
    num_heads = 8

    query = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    key = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    value = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    rot_pos_emb = keras.ops.convert_to_tensor(
        np.random.randn(seq_len, embedding_dim).astype("float32")
    )

    attention = MoonshineCausalMultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
    )

    # First call without cache.
    output, key_cache, value_cache = attention(
        query=query,
        key=key,
        value=value,
        rot_pos_emb=rot_pos_emb,
    )

    # Subsequent call with cache.
    output, key_cache_updated, value_cache_updated = attention(
        query=query,
        key=key,
        value=value,
        rot_pos_emb=rot_pos_emb,
        key_cache=key_cache,
        value_cache=value_cache,
    )
    print(output)
    ```
    """

    def build(self, query_shape, value_shape, key_shape=None):
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

        # Build attention computation sublayer.
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

    def _compute_causal_mask(self, query, value=None, for_cache=False):
        if for_cache:
            assert value is not None
            v_seq_length = keras.ops.shape(value)[1]
        else:
            v_seq_length = keras.ops.shape(query)[1]
        q_seq_length = keras.ops.shape(query)[1]
        n_rows = v_seq_length if for_cache else q_seq_length
        ones_mask = keras.ops.ones((1, n_rows, v_seq_length), dtype="int32")
        row_index = keras.ops.cumsum(ones_mask, axis=-2)
        col_index = keras.ops.cumsum(ones_mask, axis=-1)
        mask = keras.ops.greater_equal(row_index, col_index)

        if for_cache:
            mask = mask[:, -q_seq_length:, :]

        return mask

    def call(
        self,
        query,
        value,
        key,
        rot_pos_emb,
        value_cache=None,
        key_cache=None,
        training=None,
        **kwargs,
    ):
        # Project inputs.
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)
        query = _apply_rotary_pos_emb(query, rot_pos_emb)
        key = _apply_rotary_pos_emb(key, rot_pos_emb)

        # Handle caching.
        if value_cache is not None:
            if key_cache is None:
                raise ValueError(
                    "key_cache should not be None when value_cache is not None"
                )
            key = keras.ops.concatenate((key_cache, key), axis=-3)
            value = keras.ops.concatenate((value_cache, value), axis=-3)

        causal_mask = self._compute_causal_mask(
            query, value, for_cache=value_cache is not None
        )

        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=causal_mask,
            training=training,
        )

        output = self._output_dense(attention_output)
        return output, key, value


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshinePrecomputedKVMultiHeadAttention(CachedMultiHeadAttention):
    """A Multi-Head Attention layer with precomputed key and value caches.

    A variant of Keras's MultiHeadAttention that supports using precomputed
    key and value tensors from a cache. This optimization is particularly
    useful in decoder architectures where keys and values can be reused
    across attention computations.

    Args:
        Inherits arguments from `keras_hub.layers.CachedMultiHeadAttention`.
        key_cache: Optional precomputed key tensor.
        value_cache: Optional precomputed value tensor.

    Returns:
        If key_cache and value_cache are None:
            A tuple of (attention_output, key_cache, value_cache)
        If key_cache and value_cache are provided:
            attention_output only

    Example:

    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
        MoonshinePrecomputedKVMultiHeadAttention
    )

    batch_size = 2
    seq_len = 10
    embedding_dim = 64
    num_heads = 8

    query = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    key = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )
    value = keras.ops.convert_to_tensor(
        np.random.randn(batch_size, seq_len, embedding_dim).astype("float32")
    )

    attention = MoonshinePrecomputedKVMultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
    )

    # First call without cache.
    output, key_cache, value_cache = attention(
        query=query,
        key=key,
        value=value,
    )

    # Subsequent call with cache.
    output = attention(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
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

    def call(
        self,
        query,
        value,
        key,
        key_cache=None,
        value_cache=None,
        training=None,
        **kwargs,
    ):
        # Project query.
        query = self._query_dense(query)

        if key_cache is None:
            if value_cache is not None:
                raise ValueError("Both key and value cache must be None")
            # Project key and value only when no cache is provided.
            key = self._key_dense(key)
            value = self._value_dense(value)
        else:
            # Use cached projections.
            key = key_cache
            value = value_cache

        # Compute attention.
        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            training=training,
        )

        output = self._output_dense(attention_output)

        if key_cache is None:
            return output, key, value
        return output
