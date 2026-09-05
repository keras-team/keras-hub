import keras
import numpy as np
from keras import ops


class T5MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention with T5 relative position bias and cache support.

    This layer implements the attention variant used by T5, where positional
    information is injected as a learned bias added to the attention scores
    rather than as a position embedding. Only the first layer of an encoder or
    decoder stack computes the bias; the remaining layers receive it through
    the `position_bias` call argument.

    The layer can be used for both self-attention and cross-attention, and
    supports a static key/value cache for autoregressive decoding. The forward
    pass can happen in one of three modes:

    - No cache (`cache` is `None`), same as regular multi-head attention.
    - Static cache (`cache` is set, `cache_update_index` is `None`). The cached
      key/value projections are used as is and the inputs are ignored. This is
      how cross-attention reuses the encoder projections during generation.
    - Updated cache (`cache` and `cache_update_index` are both set). New
      key/value projections are computed from the inputs and spliced into the
      cache at `cache_update_index`.

    Note that caching is useful only during inference and should not be used
    during training.

    Args:
        is_decoder: bool. Whether this layer belongs to the decoder stack. The
            relative position bias is unidirectional for the decoder and
            bidirectional for the encoder.
        hidden_dim: int. The size of the layer output.
        key_value_dim: int. The size of each attention head.
        num_heads: int. The number of attention heads.
        dropout: float. Dropout probability applied to the attention weights.
        use_relative_attention_bias: bool. Whether this layer owns the relative
            attention bias weights. Only the first layer of a stack should set
            this to `True`.

    Call arguments:
        mask: Optional mask of shape
            `(batch_size, target_length, source_length)`, where a `1` marks a
            position that can be attended to. It is folded into the returned
            `position_bias`, so it is ignored when `position_bias` is passed.
        key_value_states: Optional key/value input of shape
            `(batch_size, source_length, dim)`, which makes this a
            cross-attention layer. When `None`, the layer attends over
            `hidden_states`.
        position_bias: Optional bias of shape
            `(batch_size, num_heads, target_length, source_length)` returned by
            an earlier layer of the same stack, reused as is.
        cache: Optional key/value cache of shape
            `(batch_size, 2, source_length, num_heads, key_value_dim)`.
        cache_update_index: Optional int or int tensor, the index along the
            sequence axis at which to update `cache`.

    Returns:
        An `(attention_output, position_bias, cache)` tuple. `cache` is `None`
        when no cache was passed in.
    """

    # This layer is adapted from Hugging Face
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_tf_t5.py
    def __init__(
        self,
        is_decoder,
        hidden_dim,
        key_value_dim,
        num_heads,
        dropout,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_decoder = is_decoder
        self.hidden_dim = hidden_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.use_relative_attention_bias = use_relative_attention_bias

        self.inner_dim = self.num_heads * self.key_value_dim
        self.relative_attention_buckets = 32
        self.relative_attention_max_distance = 128

        self.query_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=(self.inner_dim * self.key_value_dim) ** -0.5
            ),
            dtype=self.dtype_policy,
            name="query_projector",
        )
        self.key_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="key_projector",
        )
        self.value_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="value_projector",
        )
        self.output_projector = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="output_projector",
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

        if self.use_relative_attention_bias:
            self.relative_attention_bias = self.add_weight(
                name="embeddings",
                shape=[self.relative_attention_buckets, self.num_heads],
                initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.inner_dim**-0.5
                ),
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """Adapted from Mesh Tensorflow.

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the
        attended-to position. If bidirectional=False, then positive relative
        positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute
        relative_positions. All relative positions >= max_distance map to
        the same bucket. All relative positions <= -max_distance map to
        the same bucket. This should allow for more graceful generalization to
        longer sequences than the model has been trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            Tensor with the same shape as relative_position,
            containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                ops.cast(
                    ops.greater(relative_position, 0),
                    dtype=relative_position.dtype,
                )
                * num_buckets
            )
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = ops.less(relative_position, max_exact)
        relative_position_if_large = max_exact + ops.cast(
            ops.log(
                ops.cast(relative_position, "float32")
                / ops.cast(max_exact, "float32")
            )
            / ops.cast(ops.log(max_distance / max_exact), "float32")
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = ops.minimum(
            relative_position_if_large, num_buckets - 1
        )
        relative_buckets += ops.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, query_start_index=0):
        """Compute binned relative position bias.

        `query_start_index` is the absolute position of the first query. It is
        non-zero during cached decoding, where the queries are a short slice
        of a longer sequence and the keys span the whole cache.
        """
        context_position = ops.arange(query_length)[:, None] + query_start_index
        memory_position = ops.arange(key_length)[None, :]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = ops.take(
            self.relative_attention_bias, relative_position_bucket, axis=0
        )  # shape (query_length, key_length, num_heads)
        values = ops.expand_dims(
            ops.transpose(values, axes=(2, 0, 1)), axis=0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        cache=None,
        cache_update_index=None,
        training=False,
    ):
        # Input is (batch_size, query_length, dim).
        batch_size, seq_length = ops.shape(hidden_states)[:2]

        def shape(states):
            # (batch_size, length, dim) -> (batch_size, length, heads, head_dim)
            return ops.reshape(
                states,
                (batch_size, -1, self.num_heads, self.key_value_dim),
            )

        query_states = shape(self.query_projector(hidden_states))

        # Self-attention projects the inputs; cross-attention projects the
        # encoder states passed as `key_value_states`.
        key_value_inputs = (
            hidden_states if key_value_states is None else key_value_states
        )
        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key_states = key_cache
                value_states = value_cache
            else:
                key_update = shape(self.key_projector(key_value_inputs))
                value_update = shape(self.value_projector(key_value_inputs))
                start = [0, cache_update_index, 0, 0]
                key_states = ops.slice_update(key_cache, start, key_update)
                value_states = ops.slice_update(
                    value_cache, start, value_update
                )
                cache = ops.stack((key_states, value_states), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key_states = shape(self.key_projector(key_value_inputs))
            value_states = shape(self.value_projector(key_value_inputs))

        key_length = ops.shape(key_states)[1]

        scores = ops.einsum(
            "bqnd,bknd->bnqk", query_states, key_states
        )  # (batch_size, num_heads, query_length, key_length)

        if position_bias is None:
            if not self.use_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.num_heads, seq_length, key_length),
                    self.compute_dtype,
                )
            else:
                # During cached decoding the queries are a slice of a longer
                # sequence, so the bias rows start at the cache index.
                position_bias = self.compute_bias(
                    seq_length,
                    key_length,
                    query_start_index=(
                        0 if cache_update_index is None else cache_update_index
                    ),
                )

            if mask is not None:
                # Add a new mask axis for the head dim.
                mask = mask[:, np.newaxis, :, :]
                # Add a very large negative position bias for masked positions.
                mask = (1.0 - ops.cast(mask, position_bias.dtype)) * -1e9
                position_bias = position_bias + mask

        scores += ops.cast(position_bias, scores.dtype)
        weights = ops.nn.softmax(
            scores, axis=-1
        )  # (batch_size, num_heads, query_length, key_length)
        weights = self.dropout_layer(
            weights, training=training
        )  # (batch_size, num_heads, query_length, key_length)

        attention_output = ops.einsum(
            "bnqk,bknd->bqnd", weights, value_states
        )  # (batch_size, query_length, num_heads, head_dim)
        attention_output = ops.reshape(
            attention_output, (batch_size, -1, self.inner_dim)
        )
        attention_output = self.output_projector(attention_output)
        return (attention_output, position_bias, cache)
