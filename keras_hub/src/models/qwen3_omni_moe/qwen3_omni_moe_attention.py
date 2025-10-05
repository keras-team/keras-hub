import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_layernorm import Qwen3OmniMoeLayerNorm


@keras_hub_export("keras_hub.models.Qwen3OmniMoeAttention")
class Qwen3OmniMoeAttention(keras.layers.Layer):
    """Multi-head attention for Qwen3-Omni MoE model."""

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        head_dim,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        sliding_window_size=4096,
        max_sequence_length=32768,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim if head_dim is not None else hidden_dim // num_query_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.sliding_window_size = sliding_window_size
        self.max_sequence_length = max_sequence_length

        # Query projection
        self.query_projection = keras.layers.Dense(
            num_query_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            name="query_projection",
        )

        # Key projection
        self.key_projection = keras.layers.Dense(
            num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            name="key_projection",
        )

        # Value projection
        self.value_projection = keras.layers.Dense(
            num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            name="value_projection",
        )

        # Output projection
        self.output_projection = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="output_projection",
        )

        # Rotary embedding
        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=10000,
            scaling_factor=1.0,
            dtype=dtype,
            name="rotary_embedding",
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        batch_size, seq_len, hidden_dim = ops.shape(hidden_states)

        # Project to query, key, value
        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        # Reshape for multi-head attention
        query = ops.reshape(
            query, (batch_size, seq_len, self.num_query_heads, self.head_dim)
        )
        key = ops.reshape(
            key, (batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        )
        value = ops.reshape(
            value, (batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        )

        # Apply rotary embedding
        if position_ids is not None:
            query = self.rotary_embedding(query, position_ids)
            key = self.rotary_embedding(key, position_ids)

        # Handle cache
        if cache is not None:
            if cache_update_index is not None:
                # Update cache
                key = ops.concatenate([cache["key"], key], axis=1)
                value = ops.concatenate([cache["value"], value], axis=1)
            else:
                # Use cache
                key = cache["key"]
                value = cache["value"]

        # Update cache
        new_cache = {
            "key": key,
            "value": value,
        }

        # Transpose for attention
        query = ops.transpose(query, (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, head_dim)
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        # Handle grouped query attention (GQA)
        # Repeat key and value for grouped query attention
        if self.num_key_value_heads < self.num_query_heads:
            num_groups = self.num_query_heads // self.num_key_value_heads
            key = ops.repeat(key, num_groups, axis=1)
            value = ops.repeat(value, num_groups, axis=1)

        # Compute attention scores
        attention_scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attention_scores = attention_scores / ops.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                # Convert 2D mask to 4D for broadcasting
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.expand_dims(attention_mask, axis=1)
            attention_scores = ops.where(
                attention_mask, attention_scores, ops.full_like(attention_scores, -1e9)
            )

        # Apply softmax
        attention_weights = ops.softmax(attention_scores, axis=-1)

        # Apply attention to values
        attention_output = ops.matmul(attention_weights, value)

        # Transpose back
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))

        # Reshape and project
        attention_output = ops.reshape(
            attention_output, (batch_size, seq_len, self.num_query_heads * self.head_dim)
        )
        attention_output = self.output_projection(attention_output)

        return {
            "hidden_states": attention_output,
            "cache": new_cache,
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "head_dim": self.head_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "sliding_window_size": self.sliding_window_size,
                "max_sequence_length": self.max_sequence_length,
            }
        )
        return config
