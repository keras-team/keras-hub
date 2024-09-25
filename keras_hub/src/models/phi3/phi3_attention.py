import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.phi3.phi3_rotary_embedding import (
    Phi3SuScaledRotaryEmbedding,
)
from keras_hub.src.utils.keras_utils import clone_initializer


class Phi3Attention(keras.layers.Layer):
    """A cached grounded query attention layer."""

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        kernel_initializer="glorot_uniform",
        dropout=0,
        max_sequence_length=4096,
        pretraining_sequence_length=4096,
        rope_max_wavelength=10000,
        rope_scaling_type=None,
        rope_scaling_short_factor=None,
        rope_scaling_long_factor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.dropout = dropout

        self.max_sequence_length = max_sequence_length
        self.pretraining_sequence_length = pretraining_sequence_length
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_short_factor = rope_scaling_short_factor
        self.rope_scaling_long_factor = rope_scaling_long_factor

        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        # Einsum variables:
        # b = batch size
        # q = query length
        # k = key/value length
        # m = model dim
        # u = num query heads
        # v = num key/value heads
        # h = head dim
        hidden_dim = inputs_shape[-1]
        head_dim = hidden_dim // self.num_query_heads
        self._norm_factor = ops.sqrt(ops.cast(head_dim, self.compute_dtype))

        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(
                None,
                self.num_key_value_heads,
                head_dim,
            ),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self.softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self.output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build((None, None, self.num_query_heads, head_dim))

        if self.rope_scaling_type is None:
            self.rotary_embedding_layer = RotaryEmbedding(
                max_wavelength=self.rope_max_wavelength,
                dtype=self.dtype_policy,
            )
        elif self.rope_scaling_type == "su":
            if len(self.rope_scaling_short_factor) != head_dim // 2:
                raise ValueError(
                    "`rope_scaling_short_factor` must be of length "
                    "`hidden_dim//num_query_heads//2`. "
                    "`len(rope_scaling_short_factor)` is "
                    f"{len(self.rope_scaling_short_factor)} "
                    f"while it should be {head_dim // 2}."
                )
            if len(self.rope_scaling_long_factor) != head_dim // 2:
                raise ValueError(
                    "`rope_scaling_long_factor` must be of length "
                    "`hidden_dim//num_query_heads//2`. "
                    "`len(rope_scaling_long_factor)` is "
                    f"{len(self.rope_scaling_long_factor)} "
                    f"while it should be {head_dim // 2}."
                )
            self.rotary_embedding_layer = Phi3SuScaledRotaryEmbedding(
                inverese_freq_short_factor=self.rope_scaling_short_factor,
                inverese_freq_long_factor=self.rope_scaling_long_factor,
                max_sequence_length=self.max_sequence_length,
                pretraining_sequence_length=self.pretraining_sequence_length,
                max_wavelength=self.rope_max_wavelength,
                dtype=self.dtype_policy,
            )
        else:
            raise ValueError(
                '`rope_scaling_type` must be `None` or `"su"`.'
                "if `None` is choosed, `RotaryEmbedding` will be used."
                'if `"su"` is choosed, `Phi3SuScaledRotaryEmbedding` will be '
                "used."
            )

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )

        query = self.query_dense(hidden_states)
        key = self.key_dense(hidden_states)
        value = self.value_dense(hidden_states)

        # Compute RoPE for queries
        query = self.rotary_embedding_layer(query, start_index=start_index)
        key = self.rotary_embedding_layer(key, start_index=start_index)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key)
                value = ops.slice_update(value_cache, start, value)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )

        # [batch_shape, seq_len, num_key_value_heads, head_dim]
        # -> [batch_shape, seq_len, num_heads, head_dim]
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attention_output = self._compute_attention(
            query, key, value, attention_mask
        )

        attention_output = self.dropout_layer(
            attention_output, training=training
        )

        attention_output = self.output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            return self.softmax(attention_scores, attention_mask[:, None, :, :])
        return self.softmax(attention_scores)

    def _compute_attention(self, query, key, value, attention_mask=None):
        attention_scores = ops.einsum("bquh,bkuh->buqk", query, key)
        attention_scores = attention_scores / self._norm_factor
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(
            "buqk,bkuh->bquh", attention_scores, value
        )

        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "pretraining_sequence_length": self.pretraining_sequence_length,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_type": self.rope_scaling_type,
                "rope_scaling_short_factor": self.rope_scaling_short_factor,
                "rope_scaling_long_factor": self.rope_scaling_long_factor,
            }
        )
        return config
