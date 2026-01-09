"""Cross-Attention layer for Llama 3.2 Vision."""

from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.llama.llama_layernorm import LlamaLayerNorm


@keras_hub_export("keras_hub.models.Llama3VisionCrossAttention")
class Llama3VisionCrossAttention(layers.Layer):
    """Gated Cross-Attention layer for Llama 3 Vision.

    This layer injects visual features into the language model using
    gated cross-attention, as described in the Llama 3.2 Vision architecture.
    The attention is applied at specific decoder layer positions to enable
    progressive multi-point integration of visual information.

    The gating mechanism uses a learnable tanh gate that starts at zero,
    allowing gradual integration of visual features during training.

    Args:
        hidden_dim: int. The dimension of the hidden states.
        num_heads: int. The number of attention heads.
        num_key_value_heads: int. The number of key/value heads for GQA.
            If None, defaults to num_heads (standard MHA).
        layer_norm_epsilon: float. Epsilon for layer normalization.
        dropout: float. Dropout rate for attention weights.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for layer computations and weights.
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
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout_rate = dropout
        self.head_dim = hidden_dim // num_heads

        # Pre-normalization for the query (text hidden states)
        self.query_norm = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="query_norm",
        )

        # Pre-normalization for the key/value (vision features)
        self.kv_norm = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="kv_norm",
        )

        # Query projection (from text)
        self.query_dense = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="query",
        )

        # Key projection (from vision)
        self.key_dense = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            name="key",
        )

        # Value projection (from vision)
        self.value_dense = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            name="value",
        )

        # Output projection
        self.output_dense = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="attention_output",
        )

        # Dropout
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        # Gating parameter - initialized to zero so no vision influence at start
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

        # Reshape query: (B, seq_len, num_heads, head_dim)
        query = ops.reshape(
            query, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        # Transpose: (B, num_heads, seq_len, head_dim)
        query = ops.transpose(query, (0, 2, 1, 3))

        # Reshape key: (B, kv_seq_len, num_kv_heads, head_dim)
        key = ops.reshape(
            key,
            (batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim),
        )
        # Transpose: (B, num_kv_heads, kv_seq_len, head_dim)
        key = ops.transpose(key, (0, 2, 1, 3))

        # Reshape value: (B, kv_seq_len, num_kv_heads, head_dim)
        value = ops.reshape(
            value,
            (batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim),
        )
        # Transpose: (B, num_kv_heads, kv_seq_len, head_dim)
        value = ops.transpose(value, (0, 2, 1, 3))

        # If using GQA, repeat key/value heads to match query heads
        if self.num_key_value_heads != self.num_heads:
            num_groups = self.num_heads // self.num_key_value_heads
            # Repeat along the head dimension
            key = ops.repeat(key, num_groups, axis=1)
            value = ops.repeat(value, num_groups, axis=1)

        # Scaled dot-product attention
        scale = ops.cast(self.head_dim ** (-0.5), query.dtype)
        attention_scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attention_scores = attention_scores * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for broadcasting: (B, 1, 1, kv_seq_len)
            attention_mask = ops.expand_dims(
                ops.expand_dims(attention_mask, 1), 1
            )
            # Convert to additive mask
            attention_scores = attention_scores + ops.where(
                attention_mask,
                ops.zeros_like(attention_scores),
                ops.full_like(attention_scores, float("-inf")),
            )

        # Softmax
        attention_weights = ops.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        attention_output = ops.matmul(attention_weights, value)

        # Transpose back: (B, seq_len, num_heads, head_dim)
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        # Reshape: (B, seq_len, hidden_dim)
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
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim).
                The text hidden states (query source).
            vision_features: Tensor of shape (batch, num_patches, hidden_dim).
                The vision features (key/value source).
            vision_mask: Optional bool tensor of shape (batch, num_patches).
                Mask for vision tokens (True = attend, False = mask).
            training: Boolean indicating training mode.

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim) with vision-augmented
            hidden states.
        """
        # 1. Pre-normalize
        normed_hidden = self.query_norm(hidden_states)
        normed_vision = self.kv_norm(vision_features)

        # 2. Project Q from text, K/V from vision
        query = self.query_dense(normed_hidden)
        key = self.key_dense(normed_vision)
        value = self.value_dense(normed_vision)

        # 3. Compute attention
        attention_output = self._compute_attention(
            query, key, value, attention_mask=vision_mask
        )

        # 4. Output projection
        attention_output = self.output_dense(attention_output)

        # 5. Gated residual connection
        # tanh(gate) starts near 0, so vision influence is minimal initially
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
