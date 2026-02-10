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
        intermediate_dim,
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
        self.intermediate_dim = intermediate_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout_rate = dropout
        self.head_dim = hidden_dim // num_heads

        # === Layers ===
        # Input layer norm applied BEFORE cross-attention (HF has this)
        self.input_layernorm = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="input_layernorm",
        )
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

        # === MLP layers (same as in HF CrossAttentionDecoderLayer) ===
        self.post_attention_layernorm = LlamaLayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="post_attention_layernorm",
        )
        self.mlp_gate_proj = layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=dtype,
            name="mlp_gate",
        )
        self.mlp_up_proj = layers.Dense(
            intermediate_dim,
            use_bias=False,
            dtype=dtype,
            name="mlp_up",
        )
        self.mlp_down_proj = layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="mlp_down",
        )

    def build(self, input_shape):
        # Pre-build norms with head_dim to match HF's per-head normalization
        # HF: q_norm and k_norm have shape (head_dim,) = (128,)
        self.query_norm.build((None, None, self.head_dim))
        self.kv_norm.build((None, None, self.head_dim))

        self.gate = self.add_weight(
            name="gate",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.mlp_gate = self.add_weight(
            name="mlp_gate",
            shape=(1,),
            initializer="zeros",
            trainable=True,
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
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]
        kv_seq_len = ops.shape(vision_features)[1]

        # Apply input layer norm BEFORE cross-attention (HF architecture)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Project
        query = self.query_dense(hidden_states)
        key = self.key_dense(vision_features)
        value = self.value_dense(vision_features)

        # Reshape to (batch, seq, num_heads, head_dim)
        query = ops.reshape(
            query, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        key = ops.reshape(
            key,
            (batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim),
        )
        value = ops.reshape(
            value,
            (batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim),
        )

        # Apply per-head normalization (HF: q_norm and k_norm on head_dim)
        query = self.query_norm(query)
        key = self.kv_norm(key)

        # Transpose for attention: (batch, heads, seq, head_dim)
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.transpose(key, (0, 2, 1, 3))
        value = ops.transpose(value, (0, 2, 1, 3))

        # Handle GQA
        if self.num_key_value_heads != self.num_heads:
            num_groups = self.num_heads // self.num_key_value_heads
            key = ops.repeat(key, num_groups, axis=1)
            value = ops.repeat(value, num_groups, axis=1)

        # Compute attention
        scale = ops.cast(self.head_dim ** (-0.5), query.dtype)
        attention_scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attention_scores = attention_scores * scale

        if vision_mask is not None:
            vision_mask = ops.expand_dims(ops.expand_dims(vision_mask, 1), 1)
            attention_scores = attention_scores + ops.where(
                ops.cast(vision_mask, "bool"),
                ops.zeros_like(attention_scores),
                ops.cast(-1e9, attention_scores.dtype)
                * ops.ones_like(attention_scores),
            )

        attention_weights = ops.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = ops.matmul(attention_weights, value)

        # Reshape back
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(
            attention_output, (batch_size, seq_len, self.hidden_dim)
        )
        attention_output = self.output_dense(attention_output)

        # Apply gated attention residual
        # (use residual from before input_layernorm)
        attn_gate_value = ops.tanh(self.gate)
        hidden_states = residual + attn_gate_value * attention_output

        # === MLP block (matches HF MllamaCrossAttentionDecoderLayer) ===
        mlp_residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # SwiGLU activation: gate * silu(gate) * up
        gate_output = self.mlp_gate_proj(hidden_states)
        up_output = self.mlp_up_proj(hidden_states)
        hidden_states = ops.silu(gate_output) * up_output
        hidden_states = self.mlp_down_proj(hidden_states)

        # Apply gated MLP residual
        mlp_gate_value = ops.tanh(self.mlp_gate)
        output = mlp_residual + mlp_gate_value * hidden_states

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "num_key_value_heads": self.num_key_value_heads,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout_rate,
            }
        )
        return config
