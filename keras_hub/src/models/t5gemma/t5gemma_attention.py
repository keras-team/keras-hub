import keras

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gemma.gemma_attention import CachedGemmaAttention
from keras_hub.src.models.t5gemma.t5gemma_layers import (
    t5gemma_kernel_initializer,
)
from keras_hub.src.utils.keras_utils import clone_initializer


def repeat_kv(hidden_states, n_rep):
    """Repeats the key/value hidden states to match the number of query heads
    for Grouped Query Attention (GQA).

    This function is used in `T5GemmaSelfAttention` and `T5GemmaCrossAttention`
    to broadcast key and value states across multiple query heads when Grouped
    Query Attention (GQA) is used (i.e., when `num_query_heads` >
    `num_key_value_heads`).

    Args:
        hidden_states: Tensor, The key or value hidden states with shape
            `(batch, num_key_value_heads, sequence_length, head_dim)`.
        n_rep: int, The number of times to repeat the key/value heads. This is
            typically `num_query_heads // num_key_value_heads`.

    Returns:
        Tensor: The expanded key/value hidden states with shape
            `(batch, num_query_heads, sequence_length, head_dim)`.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = keras.ops.shape(hidden_states)
    hidden_states = keras.ops.expand_dims(hidden_states, 2)
    hidden_states = keras.ops.tile(hidden_states, (1, 1, n_rep, 1, 1))
    return keras.ops.reshape(
        hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim)
    )


@keras.saving.register_keras_serializable(package="keras_hub")
class T5GemmaSelfAttention(CachedGemmaAttention):
    """Self-attention block for the T5Gemma model.

    This layer performs self-attention with Rotary Positional Embeddings (RoPE)
    and supports Grouped Query Attention (GQA). It is used in
    `T5GemmaEncoderLayer` and `T5GemmaDecoderLayer`.

    Args:
        hidden_size: int, The dimensionality of the hidden states.
        num_attention_heads: int, The number of attention heads.
        num_key_value_heads: int, The number of key-value heads. For GQA, this
            can be less than `num_attention_heads`.
        query_pre_attn_scalar: float, Scalar to multiply queries by before
            attention.
        attention_bias: bool, Whether to include bias in the query, key, value,
            and output dense layers.
        initializer_range: float, The range for the random normal initializer
            for kernel weights. Default is `0.02`.
        attention_dropout: float, The dropout rate applied to attention weights.
            Default is `0.0`.
        attn_logit_softcapping: float, optional, The softcapping value for
            attention logits.
        rope_max_wavelength: float, The maximum wavelength for Rotary Positional
            Embeddings. Default is `10000.0`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        query_pre_attn_scalar,
        attention_bias,
        initializer_range=0.02,
        attention_dropout=0.0,
        attn_logit_softcapping=None,
        rope_max_wavelength=10000.0,
        **kwargs,
    ):
        super().__init__(
            head_dim=hidden_size // num_attention_heads,
            num_query_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            kernel_initializer=t5gemma_kernel_initializer(initializer_range),
            logit_soft_cap=attn_logit_softcapping,
            dropout=attention_dropout,
            query_head_dim_normalize=False,
            use_sliding_window_attention=False,
            **kwargs,
        )
        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.num_key_value_groups = (
            self.num_query_heads // self.num_key_value_heads
        )
        self.scaling = self.query_pre_attn_scalar**-0.5
        self.rope_max_wavelength = rope_max_wavelength
        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            sequence_axis=2,
            feature_axis=3,
            name="rotary_embedding",
        )

    def build(self, input_shape):
        self._kernel_initializer = t5gemma_kernel_initializer(
            self.initializer_range
        )

        # Query projection layer.
        self.hidden_dim = input_shape[-1]
        self.query_dense = keras.layers.EinsumDense(
            equation="...a,abc->...bc",
            output_shape=(self.num_query_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="bc" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(input_shape)

        # Key projection layer.
        self.key_dense = keras.layers.EinsumDense(
            equation="...a,abc->...bc",
            output_shape=(self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="bc" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(input_shape)

        # Value projection layer.
        self.value_dense = keras.layers.EinsumDense(
            equation="...a,abc->...bc",
            output_shape=(self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="bc" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(input_shape)

        # Output projection layer.
        self.output_dense = keras.layers.EinsumDense(
            equation="...a,ab->...b",
            output_shape=(self.hidden_dim,),
            kernel_initializer=clone_initializer(self._kernel_initializer),
            bias_axes="b" if self.attention_bias else None,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (*input_shape[:-1], self.num_query_heads * self.head_dim)
        )
        self.dropout_layer = keras.layers.Dropout(
            rate=self.attention_dropout,
            dtype=self.dtype_policy,
        )
        q_len = input_shape[1]
        attn_weights_shape = (None, self.num_query_heads, q_len, q_len)
        self.dropout_layer.build(attn_weights_shape)
        self.softmax = keras.layers.Softmax(dtype="float32")
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        query_states = self.query_dense(hidden_states)
        query_states = keras.ops.transpose(query_states, (0, 2, 1, 3))
        key_states = self.key_dense(hidden_states)
        key_states = keras.ops.transpose(key_states, (0, 2, 1, 3))
        value_states = self.value_dense(hidden_states)
        value_states = keras.ops.transpose(value_states, (0, 2, 1, 3))
        start_index = 0 if cache_update_index is None else cache_update_index
        query_states = self.rotary_embedding(
            query_states, start_index=start_index
        )
        key_states = self.rotary_embedding(key_states, start_index=start_index)
        current_pass_cache = keras.ops.stack((key_states, value_states), axis=1)
        if cache is not None:
            if cache_update_index is None:
                raise ValueError(
                    "Both `cache` and `cache_update_index` must be "
                    "passed for caching."
                )
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            start = [0, 0, cache_update_index, 0]
            key_states = keras.ops.slice_update(key_cache, start, key_states)
            value_states = keras.ops.slice_update(
                value_cache, start, value_states
            )
            cache = keras.ops.stack((key_states, value_states), axis=1)
        elif cache_update_index is not None:
            raise ValueError(
                "`cache_update_index` should not be set if `cache` is `None`."
            )
        else:
            cache = current_pass_cache

        # Repeat key-value heads for GQA.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            keras.ops.matmul(
                query_states, keras.ops.transpose(key_states, (0, 1, 3, 2))
            )
            * self.scaling
        )

        if self.logit_soft_cap is not None:
            attn_weights = attn_weights / self.logit_soft_cap
            attn_weights = keras.ops.tanh(attn_weights)
            attn_weights = attn_weights * self.logit_soft_cap
        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = keras.ops.cast(
            self.softmax(attn_weights),
            query_states.dtype,
        )
        attn_weights = self.dropout_layer(attn_weights, training=training)
        attn_output = keras.ops.matmul(attn_weights, value_states)
        attn_output = keras.ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = keras.ops.reshape(
            attn_output,
            (
                keras.ops.shape(hidden_states)[0],
                -1,
                self.num_query_heads * self.head_dim,
            ),
        )
        attn_output = self.output_dense(attn_output)
        return (attn_output, attn_weights), cache

    def compute_output_shape(self, input_shape):
        attn_output_shape = input_shape
        q_len = input_shape[1]
        attn_weights_shape = (
            input_shape[0],
            self.num_query_heads,
            q_len,
            q_len,
        )
        return attn_output_shape, attn_weights_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "attention_bias": self.attention_bias,
                "initializer_range": self.initializer_range,
                "attention_dropout": self.attention_dropout,
                "attn_logit_softcapping": self.logit_soft_cap,
                "rope_max_wavelength": self.rope_max_wavelength,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class T5GemmaCrossAttention(keras.layers.Layer):
    """Cross-attention block for the T5Gemma model.

    This layer performs cross-attention, where queries are derived from the
    decoder hidden states and keys/values are from the encoder hidden states.
    It supports Grouped Query Attention (GQA). It is used in
    `T5GemmaDecoderLayer`.

    Args:
        hidden_size: int, The dimensionality of the hidden states for queries
            and output.
        cross_attention_hidden_size: int, The dimensionality of the hidden
            states from the encoder for keys and values.
        num_attention_heads: int, The number of attention heads for queries.
        num_key_value_heads: int, The number of key-value heads. For GQA, this
            can be less than `num_attention_heads`.
        query_pre_attn_scalar: float, Scalar to multiply queries by before
            attention.
        attention_bias: bool, Whether to include bias in the query, key, value,
            and output dense layers.
        initializer_range: float, The range for the random normal initializer
            for kernel weights. Default is `0.02`.
        attention_dropout: float, The dropout rate applied to attention weights.
            Default is `0.0`.
        attn_logit_softcapping: float, optional, The softcapping value for
            attention logits.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        hidden_size,
        cross_attention_hidden_size,
        num_attention_heads,
        num_key_value_heads,
        query_pre_attn_scalar,
        attention_bias,
        initializer_range=0.02,
        attention_dropout=0.0,
        attn_logit_softcapping=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.cross_attention_hidden_size = (
            cross_attention_hidden_size or hidden_size
        )
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.attn_logit_softcapping = attn_logit_softcapping

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.query_pre_attn_scalar**-0.5

    def build(self, input_shape):
        hidden_states_shape, encoder_hidden_states_shape = input_shape
        self.kernel_initializer = t5gemma_kernel_initializer(
            self.initializer_range
        )

        # Query projection layer.
        self.query_dense = keras.layers.EinsumDense(
            equation="...a,abc->...bc",
            output_shape=(self.num_attention_heads, self.head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_axes="bc" if self.attention_bias else None,
            name="query",
        )
        self.query_dense.build(hidden_states_shape)
        cross_attn_proj_shape = (
            *encoder_hidden_states_shape[:-1],
            self.cross_attention_hidden_size,
        )

        # Key projection layer.
        self.key_dense = keras.layers.EinsumDense(
            equation="...a,abc->...bc",
            output_shape=(self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_axes="bc" if self.attention_bias else None,
            name="key",
        )
        self.key_dense.build(cross_attn_proj_shape)

        # Value projection layer.
        self.value_dense = keras.layers.EinsumDense(
            equation="...a,abc->...bc",
            output_shape=(self.num_key_value_heads, self.head_dim),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_axes="bc" if self.attention_bias else None,
            name="value",
        )
        self.value_dense.build(cross_attn_proj_shape)

        # Output projection layer.
        self.output_dense = keras.layers.EinsumDense(
            equation="...a,ab->...b",
            output_shape=(self.hidden_size,),
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_axes="b" if self.attention_bias else None,
            name="attention_output",
        )
        o_proj_input_shape = (*hidden_states_shape[:-1], self.hidden_size)
        self.output_dense.build(o_proj_input_shape)
        self.dropout_layer = keras.layers.Dropout(self.attention_dropout)
        q_len = hidden_states_shape[1]
        kv_len = encoder_hidden_states_shape[1]
        attn_weights_shape = (None, self.num_attention_heads, q_len, kv_len)
        self.dropout_layer.build(attn_weights_shape)
        self.softmax = keras.layers.Softmax(dtype="float32")
        self.built = True

    def call(
        self,
        inputs,
        attention_mask=None,
        cache=None,
        training=None,
    ):
        hidden_states, encoder_hidden_states = inputs
        batch_size, q_seq_len = keras.ops.shape(hidden_states)[:2]
        query_states = self.query_dense(hidden_states)
        query_states = keras.ops.transpose(query_states, (0, 2, 1, 3))
        if cache is not None:
            key_states = cache[:, 0, ...]
            value_states = cache[:, 1, ...]
        else:
            key_states = self.key_dense(encoder_hidden_states)
            key_states = keras.ops.transpose(key_states, (0, 2, 1, 3))
            value_states = self.value_dense(encoder_hidden_states)
            value_states = keras.ops.transpose(value_states, (0, 2, 1, 3))

        # Repeat key-value heads for GQA.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            keras.ops.matmul(
                query_states, keras.ops.transpose(key_states, (0, 1, 3, 2))
            )
            * self.scaling
        )

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = keras.ops.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping
        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = keras.ops.cast(
            self.softmax(attn_weights),
            query_states.dtype,
        )
        attn_weights = self.dropout_layer(attn_weights, training=training)
        attn_output = keras.ops.matmul(attn_weights, value_states)
        attn_output = keras.ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = keras.ops.reshape(
            attn_output, (batch_size, q_seq_len, -1)
        )
        attn_output = self.output_dense(attn_output)
        if cache is not None:
            updated_cache = keras.ops.stack((key_states, value_states), axis=1)
            return (attn_output, attn_weights), updated_cache
        else:
            return attn_output, attn_weights

    def compute_output_shape(self, input_shape):
        hidden_states_shape, encoder_hidden_states_shape = input_shape
        attn_output_shape = hidden_states_shape
        q_len = hidden_states_shape[1]
        kv_len = encoder_hidden_states_shape[1]
        attn_weights_shape = (
            hidden_states_shape[0],
            self.num_attention_heads,
            q_len,
            kv_len,
        )
        return attn_output_shape, attn_weights_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "cross_attention_hidden_size": self.cross_attention_hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "query_pre_attn_scalar": self.query_pre_attn_scalar,
                "attention_bias": self.attention_bias,
                "initializer_range": self.initializer_range,
                "attention_dropout": self.attention_dropout,
                "attn_logit_softcapping": self.attn_logit_softcapping,
            }
        )
        return config
