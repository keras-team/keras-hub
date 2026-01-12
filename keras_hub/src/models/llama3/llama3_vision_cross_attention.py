from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama.llama_layernorm import LlamaLayerNorm


@keras_hub_export("keras_hub.models.Llama3VisionCrossAttention")
class Llama3VisionCrossAttention(layers.Layer):
    """Gated cross-attention layer for Llama 3.2 Vision.

    This layer injects visual features into the language model using gated
    cross-attention. A learnable tanh gate initialized to zero enables
    gradual integration of visual features during training.

    Args:
        hidden_dim: int. The dimension of the hidden states.
        num_heads: int. The number of attention heads.
        num_key_value_heads: int. The number of key/value heads for GQA.
            Defaults to `num_heads`.
        layer_norm_epsilon: float. Epsilon for layer normalization.
            Defaults to `1e-6`.
        dropout: float. Dropout rate. Defaults to `0.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_key_value_heads=None,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        # === Config ===
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout_rate = dropout
        self.head_dim = hidden_dim // num_heads

        # === Layers ===
        self.query_norm = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="query_norm",
        )
        self.kv_norm = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="kv_norm",
        )
        self.query_dense = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="query",
        )
        self.key_dense = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            name="key",
        )
        self.value_dense = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            name="value",
        )
        self.output_dense = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="attention_output",
        )
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        self.gate = self.add_weight(
            name="gate",
            shape=(1,),
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
        )
        super().build(input_shape)

    def _compute_attention(self, query, key, value, attention_mask=None):
        """Compute scaled dot-product attention with optional GQA."""
        batch_size = ops.shape(query)[0]
        seq_len = ops.shape(query)[1]
        kv_seq_len = ops.shape(key)[1]

        query = ops.reshape(
            query, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        query = ops.transpose(query, (0, 2, 1, 3))

        key = ops.reshape(
            key,
            (batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim),
        )
        key = ops.transpose(key, (0, 2, 1, 3))

        value = ops.reshape(
            value,
            (batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim),
        )
        value = ops.transpose(value, (0, 2, 1, 3))

        if self.num_key_value_heads != self.num_heads:
            num_groups = self.num_heads // self.num_key_value_heads
            key = ops.repeat(key, num_groups, axis=1)
            value = ops.repeat(value, num_groups, axis=1)

        scale = ops.cast(self.head_dim ** (-0.5), query.dtype)
        attention_scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attention_scores = attention_scores * scale

        if attention_mask is not None:
            attention_mask = ops.expand_dims(
                ops.expand_dims(attention_mask, 1), 1
            )
            attention_scores = attention_scores + ops.where(
                attention_mask,
                ops.zeros_like(attention_scores),
                ops.full_like(attention_scores, float("-inf")),
            )

        attention_weights = ops.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = ops.matmul(attention_weights, value)

        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(
            attention_output, (batch_size, seq_len, self.hidden_dim)
        )
        return attention_output

    def call(
        self,
        hidden_states,
        vision_features,
        vision_mask=None,
        training=None,
    ):
        """Forward pass of the cross-attention layer.

        Args:
            hidden_states: Tensor of shape `(batch, seq_len, hidden_dim)`.
            vision_features: Tensor of shape `(batch, num_patches, hidden_dim)`.
            vision_mask: Optional bool tensor of shape `(batch, num_patches)`.
            training: Boolean indicating training mode.

        Returns:
            Tensor of shape `(batch, seq_len, hidden_dim)`.
        """
        normed_hidden = self.query_norm(hidden_states)
        normed_vision = self.kv_norm(vision_features)

        query = self.query_dense(normed_hidden)
        key = self.key_dense(normed_vision)
        value = self.value_dense(normed_vision)

        attention_output = self._compute_attention(
            query, key, value, attention_mask=vision_mask
        )
        attention_output = self.output_dense(attention_output)

        gate_value = ops.tanh(self.gate)
        output = hidden_states + gate_value * attention_output
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout_rate,
            }
        )
        return config
