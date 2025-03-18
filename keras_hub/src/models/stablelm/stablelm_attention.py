import math
import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.utils.keras_utils import has_flash_attention_support


class StableLMAttention(keras.layers.Layer):
    """StableLMAttention layer.

    This layer implements the attention mechanism for StableLM-3B4E1T, featuring
    multi-head self-attention with partial rotary position embeddings applied
    to a fraction of the head dimensions, as specified by `rotary_percentage`.
    It is adapted from the LlamaAttention layer with modifications to align
    with StableLM's official configuration(https://github.com/Stability-AI/StableLM/blob/main/configs/stablelm-3b-4e1t.yml).

    Args:
        num_query_heads (int): Number of attention heads for queries.
        num_key_value_heads (int): Number of attention heads for keys and values.
        hidden_dim (int): Hidden dimension of the input (e.g., 2560 for StableLM-3B4E1T).
        rope_max_wavelength (float): Maximum wavelength for rotary embeddings (default: 10000).
        rope_scaling_factor (float): Scaling factor for rotary embeddings (default: 1.0).
        rotary_percentage (float): Percentage of head dimensions to apply rotary embeddings (default: 0.25).
        kernel_initializer (str or initializer): Initializer for dense layer kernels (default: "glorot_uniform").
        dropout (float): Dropout rate for attention scores (default: 0.0).
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        rotary_percentage=0.25,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.rotary_percentage = rotary_percentage
        self.dropout = dropout
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        head_dim = self.hidden_dim // self.num_query_heads
        self.rotary_dim = int(head_dim * self.rotary_percentage)
        self._inv_norm_factor = 1.0 / math.sqrt(head_dim)

        # Query projection (no bias )
        self._query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self._query_dense.build(inputs_shape)

        # Key projection (no bias)
        self._key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self._key_dense.build(inputs_shape)

        # Value projection (no bias)
        self._value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, head_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self._value_dense.build(inputs_shape)

        # Softmax layer for attention scores
        self._softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        # Dropout layer for attention scores
        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        # Output projection (without bias)
        self._output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self._output_dense.build((None, None, self.num_query_heads, head_dim))

        # Rotary embedding layer
        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            dtype=self.dtype_policy,
        )

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        start_index = cache_update_index if cache_update_index is not None else 0

        # Compute query and apply partial rotary embedding
        query = self._query_dense(hidden_states)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = self.rotary_embedding_layer(query_rot, start_index=start_index)
        query = ops.concatenate([query_rot, query_pass], axis=-1)

        def _compute_key_value(x):
            key = self._key_dense(x)
            value = self._value_dense(x)
            # Apply partial rotary embedding to key
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = self.rotary_embedding_layer(key_rot, start_index=start_index)
            key = ops.concatenate([key_rot, key_pass], axis=-1)
            return key, value

        # Handle caching for key and value
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

        # Adjust key and value for grouped-query attention (if applicable)
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        # Compute attention output
        attention_output = self._compute_attention(query, key, value, attention_mask)
        attention_output = self._dropout_layer(attention_output, training=training)
        attention_output = self._output_dense(attention_output)

        return attention_output, cache if cache is not None else attention_output

    def _masked_softmax(self, attention_scores, attention_mask=None):
        if attention_mask is not None:
            return self._softmax(attention_scores, attention_mask[:, None, :, :])
        return self._softmax(attention_scores)

    def _compute_attention(self, query, key, value, attention_mask=None):
        if has_flash_attention_support() and self.dropout == 0:
            if attention_mask is not None:
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.cast(attention_mask, dtype="bool")
            return ops.dot_product_attention(
                query,
                key,
                value,
                mask=attention_mask,
                scale=self._inv_norm_factor,
            )

        attention_scores = ops.einsum(self._dot_product_equation, query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(self._combine_equation, attention_scores, value)
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rotary_percentage": self.rotary_percentage,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
            }
        )
        return config


