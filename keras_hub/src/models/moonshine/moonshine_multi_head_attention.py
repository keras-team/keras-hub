import keras
from keras import backend

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.models.whisper.whisper_cached_multi_head_attention import (
    _build_proj_equation,
)
from keras_hub.src.models.whisper.whisper_cached_multi_head_attention import (
    _get_output_shape,
)


# Removed dependence on einops.
# Source: https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L35
def _rotate_half(x):
    """
    Rotates the two halves of the last dimension.

    This function splits the last dimension of the input tensor into two equal
    halves and swaps them with a sign inversion. Specifically, for an input of
    shape `[..., 2*d]`, it returns a tensor of the same shape where `[x1, x2]`
    is transformed into `[-x2, x1]`.

    Args:
        x: Tensor. Shape `[..., 2*d]`. The input tensor to be rotated.

    Returns:
        Tensor: A tensor of shape `[..., 2*d]` with the two halves rotated.
    """
    # Conditional for Tensorflow backend.
    if backend.backend() == "tensorflow":
        x_shape = keras.ops.shape(x)
        last_dim = x_shape[-1]
        d = last_dim // 2
        x_shape_tensor = keras.ops.convert_to_tensor(x_shape)
        new_shape = keras.ops.concatenate(
            [x_shape_tensor[:-1], keras.ops.convert_to_tensor([d, 2])], axis=0
        )
        x = keras.ops.reshape(x, new_shape)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x_rotated = keras.ops.stack([-x2, x1], axis=-1)
        x_rotated = keras.ops.reshape(x_rotated, x_shape)
        return x_rotated

    # Conditional for PyTorch and JAX backends.
    if backend.backend() == "torch" or backend.backend() == "jax":
        x_shape = keras.ops.shape(x)
        x_shape_tuple = tuple(
            int(keras.ops.convert_to_numpy(dim).item()) for dim in x_shape
        )
        last_dim = x_shape_tuple[-1]
        d = last_dim // 2
        new_shape = x_shape_tuple[:-1] + (d, 2)
        x = keras.ops.reshape(x, new_shape)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x_rotated = keras.ops.stack([-x2, x1], axis=-1)
        x_rotated = keras.ops.reshape(x_rotated, x_shape_tuple)
        return x_rotated

    else:
        raise NotImplementedError(
            "Backend not supported. Please use TensorFlow, PyTorch, or JAX."
        )


def _apply_rotary_pos_emb(t, freqs):
    """
    Applies rotary positional embeddings to the input tensor. Used in on-the-fly
    computation of rotary positional embeddings in multi-head attention layers.

    Args:
        t: A tensor with shape `[..., seq_len, ..., hidden_dim]` where the
            rotary embedding is applied to the first `rot_dim` channels of the
            last dimension.
        freqs: A tensor of frequency values with shape `[max_seq_len, rot_dim]`.
            The last `seq_len` entries are used to compute the rotary
            embeddings.

    Returns:
        Tensor: A tensor of the same shape as `t` with the rotary positional
        embeddings applied to the first `rot_dim` channels of the last dimension
        and the remaining channels concatenated unchanged.
    """
    rot_dim = keras.ops.shape(freqs)[-1]
    seq_len = keras.ops.shape(t)[-3]
    orig_dtype = t.dtype
    freqs = freqs[-seq_len:, :]
    freqs = keras.ops.reshape(freqs, (seq_len, 1, rot_dim))
    t_rot = t[..., :rot_dim]
    t_nonrot = t[..., rot_dim:]
    t_rotated = t_rot * keras.ops.cos(freqs) + _rotate_half(
        t_rot
    ) * keras.ops.sin(freqs)
    out = keras.ops.concatenate([t_rotated, t_nonrot], axis=-1)
    return keras.ops.cast(out, orig_dtype)


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineMultiHeadAttention(keras.layers.MultiHeadAttention):
    """Multi-head attention with rotary positional embeddings.

    Extends `keras.layers.MultiHeadAttention` by integrating rotary positional
    embeddings (RoPE) into query and key projections using the
    `_apply_rotary_pos_emb` function. This implementation follows KerasHub's
    Whisper architecture and optimizes attention computations for each backend.

    The layer projects queries, keys, and values, applies RoPE, computes scaled
    dot-product attention, and projects the output.

    Args:
        num_heads: int. Number of attention heads.
        key_dim: int. Dimensionality of queries and keys per head.
        value_dim: int. Dimensionality of values per head.
        attention_bias: bool. Whether to use bias in the attention mechanism.
            Defaults to False.
        attention_dropout: float. Dropout rate applied to attention weights.
            Defaults to 0.0.
        use_bias: bool. Whether to use bias in the projection layers.
        output_shape: int, tuple, or list. Output dimensionality of the layer.
        attention_axes: tuple or list. Axes over which attention is applied.
        dropout: float. Dropout probability for attention weights.
        kernel_initializer: str or initializer. Initializer for projection
            kernels.
        bias_initializer: str or initializer. Initializer for bias vectors.
        kernel_regularizer: regularizer. Regularizer for projection kernels.
        bias_regularizer: regularizer. Regularizer for bias vectors.
        activity_regularizer: regularizer. Regularizer for attention outputs.
        kernel_constraint: constraint. Constraint for projection kernels.
        bias_constraint: constraint. Constraint for bias vectors.
        rotary_embedding: Tensor. Rotary positional embeddings applied to
            queries and keys.

    Returns:
        Tensor: A tensor of shape `[batch_size, seq_length, num_heads *
        value_dim]` representing the attention output.

    ## References
    Based on the [UsefulSensors implementation of MHAWithRope](https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L59)
    class.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        kwargs.pop("use_bias", None)
        kwargs.pop("dropout", None)
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            use_bias=attention_bias,
            dropout=attention_dropout,
            **kwargs,
        )

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
        self, query, value, key, rotary_embedding, training=None, **kwargs
    ):
        # Project query, key, and value.
        query_proj = self._query_dense(query)
        key_proj = self._key_dense(key)
        value_proj = self._value_dense(value)
        # Apply rotary positional embeddings to query and key.
        query_proj = _apply_rotary_pos_emb(query_proj, rotary_embedding)
        key_proj = _apply_rotary_pos_emb(key_proj, rotary_embedding)
        # Compute attention.
        attention_output, _ = self._compute_attention(
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
    """
    Causal multi-head attention with rotary positional embeddings.

    Extends `keras_hub.layers.CachedMultiHeadAttention` by integrating rotary
    positional embeddings (RoPE), causal masking, and state caching. This
    implementation follows KerasHub's Whisper architecture and optimizes
    autoregressive attention.

    Key differences from `keras_hub.layers.CachedMultiHeadAttention`:
    - Applies rotary embeddings to queries and keys via `_apply_rotary_pos_emb`.
    - Uses `_compute_causal_mask` for optimized autoregressive masking.

    Args:
        num_heads: int. Number of attention heads.
        key_dim: int. Dimensionality of queries and keys per head.
        value_dim: int. Dimensionality of values per head.
        attention_bias: bool. Whether to use bias in the attention mechanism.
            Defaults to False.
        attention_dropout: float. Dropout rate applied to attention weights.
            Defaults to 0.0.
        use_bias: bool. Whether to use bias in projection layers.
        output_shape: int, tuple, or list. Output dimensionality of the layer.
        attention_axes: tuple or list. Axes over which attention is applied.
        dropout: float. Dropout probability for attention weights.
        kernel_initializer: str or initializer. Initializer for projection
            kernels.
        bias_initializer: str or initializer. Initializer for bias vectors.
        kernel_regularizer: regularizer. Regularizer for projection kernels.
        bias_regularizer: regularizer. Regularizer for bias vectors.
        activity_regularizer: regularizer. Regularizer for attention outputs.
        kernel_constraint: constraint. Constraint for projection kernels.
        bias_constraint: constraint. Constraint for bias vectors.
        rotary_embedding: Tensor. Rotary positional embeddings applied to
            queries and keys.
        key_cache: Tensor, optional. Cached key projections from previous
            attention computations.
        value_cache: Tensor, optional. Cached value projections from previous
            attention computations.

    Returns:
        Tuple: `(attention_output, key_state, value_state)`, where:
            - attention_output: Processed attention output.
            - key_state: Updated key cache state.
            - value_state: Updated value cache state.

    ## References
    Based on the [UsefulSensors implementation of MHACausalWithRope](https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L240C7-L240C24)
    class.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        kwargs.pop("use_bias", None)
        kwargs.pop("dropout", None)
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            use_bias=attention_bias,
            dropout=attention_dropout,
            **kwargs,
        )

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

    def compute_output_spec(self, query, value, key):
        return (
            keras.KerasTensor(query.shape[:-1] + (self._value_dim,)),
            keras.KerasTensor(key.shape),
            keras.KerasTensor(value.shape),
        )

    def _compute_causal_mask(self, query, value=None, for_cache=False):
        if backend.backend() == "torch" or backend.backend() == "jax":
            q_seq_length = int(
                keras.ops.convert_to_numpy(keras.ops.shape(query)[1]).item()
            )
            v_seq_length = (
                int(
                    keras.ops.convert_to_numpy(keras.ops.shape(value)[1]).item()
                )
                if value is not None
                else q_seq_length
            )
        elif backend.backend() == "tensorflow":
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
        rotary_embedding,
        value_cache=None,
        key_cache=None,
        training=None,
        **kwargs,
    ):
        # Project inputs.
        query = self._query_dense(query)
        key = self._key_dense(key)
        value = self._value_dense(value)
        query = _apply_rotary_pos_emb(query, rotary_embedding)
        key = _apply_rotary_pos_emb(key, rotary_embedding)

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
    """
    Multi-head attention with precomputed key and value caches.

    Extends `CachedMultiHeadAttention` to support scenarios where key and value
    projections remain static, such as cross-attention in encoder-decoder
    transformers. If caches are provided, it bypasses `_key_dense` and
    `_value_dense` projections in the `call` method, optimizing performance.

    **Behavior:**
    - If no cache is supplied, returns `(attention_output, key_cache,
    value_cache)`.
    - If caches are provided, computes attention using the cached keys and
    values and returns only `attention_output`.

    Args:
        num_heads: int. Number of attention heads.
        key_dim: int. Dimensionality of queries and keys per head.
        value_dim: int. Dimensionality of values per head.
        attention_bias: bool. Whether to use bias in the attention mechanism.
            Defaults to False.
        attention_dropout: float. Dropout rate applied to attention weights.
            Defaults to 0.0.
        use_bias: bool. Whether to use bias in projection layers.
        output_shape: int, tuple, or list. Output dimensionality of the layer.
        attention_axes: tuple or list. Axes over which attention is applied.
        dropout: float. Dropout probability for attention weights.
        kernel_initializer: str or initializer. Initializer for projection
            kernels.
        bias_initializer: str or initializer. Initializer for bias vectors.
        kernel_regularizer: regularizer. Regularizer for projection kernels.
        bias_regularizer: regularizer. Regularizer for bias vectors.
        activity_regularizer: regularizer. Regularizer for attention outputs.
        kernel_constraint: constraint. Constraint for projection kernels.
        bias_constraint: constraint. Constraint for bias vectors.
        key_cache: Tensor, optional. Precomputed key projections.
        value_cache: Tensor, optional. Precomputed value projections.

    Returns:
        Tuple/Tensor: `(attention_output, key_cache, value_cache)` if no caches
            are provided, or only `attention_output` if caches are provided.

    ## References
    Based on the [UsefulSensors implementation of MHAPrecomputedKV](https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L310)
    class.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        kwargs.pop("use_bias", None)
        kwargs.pop("dropout", None)
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            use_bias=attention_bias,
            dropout=attention_dropout,
            **kwargs,
        )

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
