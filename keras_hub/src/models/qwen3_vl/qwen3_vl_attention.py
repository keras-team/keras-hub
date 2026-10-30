import math

import keras
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import Qwen3_VLLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import fused_attention_op_available


def _interleave_selector(mrope_section):
    # One-hot (3, half_dim) choosing T, H or W for each frequency index.
    half_dim = sum(mrope_section)
    channel = [0] * half_dim
    for dim in (1, 2):
        for index in range(dim, min(mrope_section[dim] * 3, half_dim), 3):
            channel[index] = dim
    channel = ops.convert_to_tensor(channel, dtype="int32")
    return ops.transpose(ops.one_hot(channel, 3, dtype="float32"))


def compute_mrope_embeddings(
    head_dim,
    rope_max_wavelength,
    mrope_section,
    seq_len,
    start_index,
    position_ids,
):
    """Return float32 `(cos, sin)` of shape `(batch or 1, seq, 1, head_dim)`.

    Every layer of a model gets the same tables, so a caller computes them
    once and passes them to each layer as `rotary_embeddings`.

    Args:
        head_dim: int. Size of an attention head.
        rope_max_wavelength: float. The base of the rotary frequencies.
        mrope_section: list of 3 ints. Frequencies used by T, H and W.
        seq_len: int. Length of the sequence.
        start_index: int. Position of the first token when `position_ids` is
            `None`.
        position_ids: Optional int Tensor `(batch, 3, seq_len)`. `None` means
            plain text positions on all three channels.
    """
    exponent = ops.arange(0, head_dim, 2, dtype="float32")
    inv_freq = ops.power(
        ops.cast(rope_max_wavelength, "float32"), -exponent / head_dim
    )
    if position_ids is None:
        positions = ops.arange(seq_len, dtype="float32") + ops.cast(
            start_index, "float32"
        )
        # (1, 1, seq): all three channels share the text position.
        positions = positions[None, None, :]
    else:
        positions = ops.cast(position_ids, "float32")
    # (batch, 3, seq, half_dim)
    freqs = positions[..., None] * inv_freq
    selector = _interleave_selector(mrope_section)
    freqs = ops.sum(freqs * selector[None, :, None, :], axis=1)
    emb = ops.concatenate([freqs, freqs], axis=-1)
    cos = ops.expand_dims(ops.cos(emb), axis=2)
    sin = ops.expand_dims(ops.sin(emb), axis=2)
    return cos, sin


class Qwen3_VLAttention(keras.layers.Layer):
    """Grouped-query attention with interleaved multimodal RoPE.

    Queries and keys are RMS-normalized per head before rotary embedding,
    as in Qwen3. Rotary frequencies are shared across all of `head_dim`, and
    the temporal, height and width position channels are interleaved over
    the frequencies (`T, H, W, T, H, W, ...`) following HF's
    `apply_interleaved_mrope`.

    Args:
        num_query_heads: int. Number of query heads.
        num_key_value_heads: int. Number of key/value heads.
        head_dim: int. Size of each attention head.
        rope_max_wavelength: float. Base of the rotary frequencies.
        mrope_section: list of 3 ints. Number of frequencies used by the
            temporal, height and width channels. Must sum to `head_dim // 2`.
        kernel_initializer: Initializer for the projection kernels.
        dropout: float. Dropout rate applied to the attention output.
        layer_norm_epsilon: float. Epsilon of the query and key RMSNorm.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        rope_max_wavelength=5000000.0,
        mrope_section=(24, 20, 20),
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sum(mrope_section) != head_dim // 2:
            raise ValueError(
                "`mrope_section` must sum to `head_dim // 2`. Received: "
                f"mrope_section={mrope_section}, head_dim={head_dim}"
            )
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.mrope_section = list(mrope_section)
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.num_key_value_groups = num_query_heads // num_key_value_heads

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)
        self._query_norm = Qwen3_VLLayerNorm(
            head_dim=self.head_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="query_norm",
        )
        self._query_norm.build((None, None, None, self.head_dim))
        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)
        self._key_norm = Qwen3_VLLayerNorm(
            head_dim=self.head_dim,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="key_norm",
        )
        self._key_norm.build((None, None, None, self.head_dim))
        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)
        self._softmax = keras.layers.Softmax(
            axis=-1, dtype="float32", name="attention_softmax"
        )
        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy
        )
        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )
        self.built = True

    def compute_rotary_embeddings(self, seq_len, start_index, position_ids):
        """Return the `(cos, sin)` tables, see `compute_mrope_embeddings`."""
        return compute_mrope_embeddings(
            self.head_dim,
            self.rope_max_wavelength,
            self.mrope_section,
            seq_len,
            start_index,
            position_ids,
        )

    def _apply_rotary(self, x, cos, sin):
        x1, x2 = ops.split(x, 2, axis=-1)
        rotated = ops.concatenate([-x2, x1], axis=-1)
        cos = ops.cast(cos, x.dtype)
        sin = ops.cast(sin, x.dtype)
        return x * cos + rotated * sin

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        position_ids=None,
        rotary_embeddings=None,
        training=None,
    ):
        """Applies attention to the hidden states.

        Args:
            hidden_states: Tensor of shape `(batch, seq_len, hidden_dim)`.
            attention_mask: Optional tensor of shape
                `(batch, seq_len, key_len)`.
            cache: Optional key/value cache of shape
                `(batch, 2, max_len, num_key_value_heads, head_dim)`.
            cache_update_index: Index at which the cache is updated.
            position_ids: Optional int Tensor of shape `(batch, 3, seq_len)`
                holding the T, H and W positions of `hidden_states`. When
                `None`, positions are `cache_update_index + arange(seq_len)`
                on all three channels.
            rotary_embeddings: Optional `(cos, sin)` tuple from
                `compute_rotary_embeddings`. Computed from `position_ids` when
                `None`.
            training: Boolean, whether in training mode.

        Returns:
            The attention output, plus the updated cache if `cache` is given.
        """
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )
        if rotary_embeddings is None:
            rotary_embeddings = self.compute_rotary_embeddings(
                ops.shape(hidden_states)[1], start_index, position_ids
            )
        cos, sin = rotary_embeddings
        query = self._query_norm(self._query_dense(hidden_states))
        query = self._apply_rotary(query, cos, sin)

        def _compute_key_value(x):
            key = self._key_norm(self._key_dense(x))
            key = ops.cast(
                self._apply_rotary(key, cos, sin), self.compute_dtype
            )
            return key, self._value_dense(x)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update, value_update = _compute_key_value(hidden_states)
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
            key, value = _compute_key_value(hidden_states)

        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query, key, value, attention_mask
        )
        attention_output = self._dropout_layer(
            attention_output, training=training
        )
        attention_output = self._output_dense(attention_output)
        if cache is not None:
            return attention_output, cache
        return attention_output

    def _compute_attention(self, query, key, value, attention_mask=None):
        if fused_attention_op_available():
            if attention_mask is not None:
                attention_mask = ops.cast(
                    ops.expand_dims(attention_mask, axis=1), "bool"
                )
            return ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )
        scores = ops.einsum("bquh,bkuh->buqk", query, key)
        scores = scores * ops.cast(self._inv_norm_factor, self.compute_dtype)
        if attention_mask is not None:
            scores = self._softmax(scores, attention_mask[:, None, :, :])
        else:
            scores = self._softmax(scores)
        scores = ops.cast(scores, self.compute_dtype)
        return ops.einsum("buqk,bkuh->bquh", scores, value)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "mrope_section": self.mrope_section,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
