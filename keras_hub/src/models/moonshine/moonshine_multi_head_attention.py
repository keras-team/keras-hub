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
    freqs = freqs[:seq_len, :]
    freqs = keras.ops.reshape(freqs, (seq_len, 1, rot_dim))
    t_rot = t[..., :rot_dim]
    t_nonrot = t[..., rot_dim:]
    t_rotated = t_rot * keras.ops.cos(freqs) + _rotate_half(
        t_rot
    ) * keras.ops.sin(freqs)
    out = keras.ops.concatenate([t_rotated, t_nonrot], axis=-1)
    return keras.ops.cast(out, orig_dtype)


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineMultiHeadAttention(CachedMultiHeadAttention):
    """
    Moonshine multi-head attention layer.

    Implements a multi-head attention mechanism for Moonshine models with
    support for rotary position embeddings and different caching strategies.
    This layer extends the `CachedMultiHeadAttention` base class to include
    specialized functionality for Moonshine models, such as rotary embeddings
    and causal masking.

    Args:
        num_heads: int. Number of attention heads.
        key_dim: int. Size of each attention head for key.
        value_dim: int, optional. Size of each attention head for value. If
            None, defaults to `key_dim`.
        attention_bias: bool, optional. Whether to include bias in attention
            projection layers. Defaults to `False`.
        attention_dropout: float, optional. Dropout probability for attention
            weights. Defaults to 0.0.
        use_causal_mask: bool, optional. Whether to apply causal masking to
            prevent positions from attending to subsequent positions. Defaults
            to `False`.
        apply_rotary_embedding: bool, optional. Whether to apply rotary position
            embeddings to queries and keys. Defaults to `True`.
        cache_mode: str, optional. Mode for key-value caching. Must be one of:
            'none': No caching.
            'autoregressive': Incremental caching for autoregressive generation.
            'precomputed': Use precomputed key-value pairs. Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    # References:
    # Based on the HuggingFace implementation of the MoonshineAttention class (https://github.com/huggingface/transformers/blob/fc8764c9a618add64c33e83720f974750bcd0978/src/transformers/models/moonshine/modeling_moonshine.py#L184-L315).

    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_causal_mask=False,
        apply_rotary_embedding=True,
        cache_mode="none",
        **kwargs,
    ):
        kwargs.pop("use_bias", None)
        kwargs.pop("dropout", None)
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            use_bias=attention_bias,
            dropout=attention_dropout,
            **kwargs,
        )
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_causal_mask = use_causal_mask
        self.apply_rotary_embedding = apply_rotary_embedding
        if cache_mode not in ["none", "autoregressive", "precomputed"]:
            raise ValueError(
                "cache_mode must be 'none', 'autoregressive', or 'precomputed'"
            )
        self.cache_mode = cache_mode

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
        output_shape = (
            query_shape[-1] if not self._output_shape else self._output_shape
        )
        if isinstance(output_shape, (list, tuple)):
            output_shape = list(output_shape)
        else:
            output_shape = [output_shape]

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
        rotary_embedding=None,
        attention_mask=None,
        key_cache=None,
        value_cache=None,
        training=None,
        **kwargs,
    ):
        # Project inputs.
        query_proj = self._query_dense(query)
        if rotary_embedding is not None:
            query_proj = _apply_rotary_pos_emb(query_proj, rotary_embedding)

        # Handle caching.
        if self.cache_mode == "none":
            key_proj = self._key_dense(key)
            value_proj = self._value_dense(value)
            if self.apply_rotary_embedding and rotary_embedding is not None:
                key_proj = _apply_rotary_pos_emb(key_proj, rotary_embedding)
            final_key = key_proj
            final_value = value_proj
        elif self.cache_mode == "autoregressive":
            if key_cache is None and value_cache is not None:
                raise ValueError(
                    "key_cache must be provided if value_cache is provided"
                )
            new_key = self._key_dense(key)
            new_value = self._value_dense(value)
            if self.apply_rotary_embedding and rotary_embedding is not None:
                new_key = _apply_rotary_pos_emb(new_key, rotary_embedding)
            if key_cache is not None and value_cache is not None:
                final_key = keras.ops.concatenate((key_cache, new_key), axis=-3)
                final_value = keras.ops.concatenate(
                    (value_cache, new_value), axis=-3
                )
            else:
                final_key = new_key
                final_value = new_value
        elif self.cache_mode == "precomputed":
            if key_cache is None and value_cache is not None:
                raise ValueError(
                    "key_cache must be provided if value_cache is provided"
                )
            if key_cache is not None and value_cache is not None:
                final_key = key_cache
                final_value = value_cache
            else:
                final_key = self._key_dense(key)
                final_value = self._value_dense(value)
        else:
            raise ValueError(f"Invalid cache_mode: {self.cache_mode}")

        # Compute attention mask.
        if self.use_causal_mask:
            causal_mask = self._compute_causal_mask(
                query,
                final_value if self.cache_mode == "autoregressive" else None,
                for_cache=(
                    self.cache_mode == "autoregressive"
                    and key_cache is not None
                ),
            )
            # Combine with attention_mask if provided.
            if attention_mask is not None:
                # [batch_size, seq_len_k] → [batch_size, 1, 1, seq_len_k].
                attention_mask_expanded = keras.ops.expand_dims(
                    attention_mask, axis=1
                )
                attention_mask_expanded = keras.ops.expand_dims(
                    attention_mask_expanded, axis=-1
                )
                final_mask = keras.ops.logical_and(
                    causal_mask, attention_mask_expanded
                )
            else:
                final_mask = causal_mask
        else:
            if attention_mask is not None:
                if self.cache_mode == "none":
                    seq_len = keras.ops.shape(query)[1]
                    final_mask = keras.ops.tile(
                        attention_mask[:, None, :], [1, seq_len, 1]
                    )
                elif self.cache_mode == "precomputed":
                    final_mask = attention_mask[:, None, None, :]
                else:  # Autoregressive
                    final_mask = attention_mask
            else:
                final_mask = None

        attention_kwargs = {
            k: v for k, v in kwargs.items() if k != "padding_mask"
        }
        # Compute attention.
        attention_output, _ = self._compute_attention(
            query=query_proj,
            key=final_key,
            value=final_value,
            attention_mask=final_mask,
            training=training,
            **attention_kwargs,
        )

        # Project the attention output.
        output = self._output_dense(attention_output)

        # Return based on cache_mode.
        if self.cache_mode == "none":
            return output
        elif self.cache_mode == "autoregressive":
            return output, final_key, final_value
        elif self.cache_mode == "precomputed":
            if key_cache is not None and value_cache is not None:
                return output
            return output, final_key, final_value
