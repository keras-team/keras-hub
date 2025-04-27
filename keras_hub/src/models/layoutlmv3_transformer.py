"""LayoutLMv3 transformer layer implementation.

This module implements the transformer layer used in the LayoutLMv3 model.
"""

from typing import Dict, Optional

from keras import backend, layers
from keras.saving import register_keras_serializable

@register_keras_serializable()
class LayoutLMv3TransformerLayer(layers.Layer):
    """Transformer layer for LayoutLMv3 model.

    This layer implements a transformer block with self-attention and feed-forward
    networks, including support for relative position embeddings.

    Args:
        hidden_size: int, defaults to 768. Size of the hidden layers.
        num_attention_heads: int, defaults to 12. Number of attention heads.
        intermediate_size: int, defaults to 3072. Size of intermediate layer.
        hidden_act: str, defaults to "gelu". Activation function for hidden layer.
        hidden_dropout_prob: float, defaults to 0.1. Dropout for hidden layers.
        attention_probs_dropout_prob: float, defaults to 0.1. Dropout for attention.
        initializer_range: float, defaults to 0.02. Initializer standard deviation.
        layer_norm_eps: float, defaults to 1e-12. Layer normalization epsilon.
        qkv_bias: bool, defaults to True. Whether to use bias in attention.
        use_rel_pos: bool, defaults to False. Whether to use relative positions.
        rel_pos_bins: int, defaults to 32. Number of relative position bins.
        max_rel_pos: int, defaults to 128. Maximum relative position distance.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
    ```python
    # Create transformer layer
    transformer = LayoutLMv3TransformerLayer(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072
    )

    # Process inputs
    outputs = transformer(inputs, attention_mask)
    ```
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_bins: int = 32,
        max_rel_pos: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos

        # Query, key, value projections
        self.q_proj = layers.Dense(hidden_size, use_bias=qkv_bias, name="attention.query")
        self.k_proj = layers.Dense(hidden_size, use_bias=qkv_bias, name="attention.key")
        self.v_proj = layers.Dense(hidden_size, use_bias=qkv_bias, name="attention.value")

        # Output projection
        self.attention_output = layers.Dense(hidden_size, name="attention.output.dense")
        self.attention_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="attention.output.LayerNorm"
        )

        # Feed-forward layers
        self.intermediate = layers.Dense(
            intermediate_size, activation=hidden_act, name="intermediate.dense"
        )
        self.output_dense = layers.Dense(hidden_size, name="output.dense")
        self.output_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="output.LayerNorm"
        )

        # Dropout
        self.dropout = layers.Dropout(hidden_dropout_prob)
        self.attention_dropout = layers.Dropout(attention_probs_dropout_prob)

        # Relative position embeddings
        if use_rel_pos:
            self.rel_pos_bias = self.add_weight(
                shape=(2 * rel_pos_bins - 1, num_attention_heads),
                initializer="zeros",
                trainable=True,
                name="rel_pos_bias",
            )

    def call(
        self, hidden_states: backend.Tensor, attention_mask: Optional[backend.Tensor] = None
    ) -> backend.Tensor:
        """Process inputs through the transformer layer.

        Args:
            hidden_states: Float tensor of shape (batch_size, seq_length, hidden_size).
                Input hidden states.
            attention_mask: Optional float tensor of shape (batch_size, 1, seq_length, seq_length).
                Attention mask where 1.0 indicates tokens to attend to and 0.0 indicates tokens to ignore.

        Returns:
            Float tensor of shape (batch_size, seq_length, hidden_size).
            The transformed hidden states.

        Example:
        ```python
        # Process sequence through transformer
        hidden_states = transformer(hidden_states, attention_mask)
        ```
        """
        batch_size = backend.shape(hidden_states)[0]
        seq_length = backend.shape(hidden_states)[1]
        head_dim = self.hidden_size // self.num_attention_heads

        # Project to query, key, value
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape and transpose for attention
        q = backend.reshape(q, (batch_size, seq_length, self.num_attention_heads, head_dim))
        k = backend.reshape(k, (batch_size, seq_length, self.num_attention_heads, head_dim))
        v = backend.reshape(v, (batch_size, seq_length, self.num_attention_heads, head_dim))

        q = backend.transpose(q, [0, 2, 1, 3])  # (batch, heads, seq_length, head_dim)
        k = backend.transpose(k, [0, 2, 1, 3])
        v = backend.transpose(v, [0, 2, 1, 3])

        # Compute attention scores
        attention_scores = backend.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / backend.sqrt(backend.cast(head_dim, "float32"))

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0

        # Apply relative position bias if enabled
        if self.use_rel_pos:
            rel_pos_bias = self._get_rel_pos_bias(seq_length)
            attention_scores = attention_scores + rel_pos_bias

        # Apply softmax and dropout
        attention_probs = backend.softmax(attention_scores, axis=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Apply attention to values
        context = backend.matmul(attention_probs, v)
        context = backend.transpose(context, [0, 2, 1, 3])  # (batch, seq_length, heads, head_dim)
        context = backend.reshape(context, (batch_size, seq_length, self.hidden_size))

        # Apply output projection and residual connection
        attention_output = self.attention_output(context)
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_layernorm(attention_output + hidden_states)

        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.output_layernorm(layer_output + attention_output)

        return layer_output

    def _get_rel_pos_bias(self, seq_length: int) -> backend.Tensor:
        """Compute relative position bias for attention scores.

        Args:
            seq_length: int. Length of input sequence.

        Returns:
            Float tensor of shape (1, num_heads, seq_length, seq_length).
            The relative position bias to be added to attention scores.
        """
        # Create relative position indices
        pos = backend.arange(seq_length, dtype="int32")
        rel_pos = pos[:, None] - pos[None, :]  # (seq_length, seq_length)
        rel_pos = rel_pos + self.rel_pos_bins - 1

        # Clip to valid range
        rel_pos = backend.clip(rel_pos, 0, 2 * self.rel_pos_bins - 2)

        # Get bias values and reshape
        bias = backend.gather(self.rel_pos_bias, rel_pos)  # (seq_length, seq_length, num_heads)
        bias = backend.transpose(bias, [2, 0, 1])  # (num_heads, seq_length, seq_length)
        bias = backend.expand_dims(bias, 0)  # (1, num_heads, seq_length, seq_length)

        return bias

    def get_config(self) -> Dict:
        """Get the layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "qkv_bias": self.qkv_bias,
            "use_rel_pos": self.use_rel_pos,
            "rel_pos_bins": self.rel_pos_bins,
            "max_rel_pos": self.max_rel_pos,
        })
        return config 