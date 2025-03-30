import keras
import tensorflow as tf
import numpy as np
from keras import layers
from keras import ops
from keras.src.saving import register_keras_serializable

@register_keras_serializable()
class LayoutLMv3Backbone(keras.Model):
    """LayoutLMv3 backbone model.
    
    This class implements the LayoutLMv3 model architecture as described in
    "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking"
    (https://arxiv.org/abs/2204.08387).
    
    Args:
        vocab_size: The size of the vocabulary.
        hidden_size: The size of the hidden layers.
        num_hidden_layers: The number of hidden layers.
        num_attention_heads: The number of attention heads.
        intermediate_size: The size of the intermediate layer in the transformer encoder.
        hidden_act: The activation function for the intermediate layer.
        hidden_dropout_prob: The dropout probability for the hidden layers.
        attention_probs_dropout_prob: The dropout probability for the attention probabilities.
        max_position_embeddings: The maximum sequence length for position embeddings.
        type_vocab_size: The size of the token type vocabulary.
        initializer_range: The standard deviation of the truncated normal initializer.
        layer_norm_eps: The epsilon value for layer normalization.
        image_size: The size of the input image (height, width).
        patch_size: The size of the image patches.
        num_channels: The number of input image channels.
        qkv_bias: Whether to use bias in the query, key, value projections.
        use_abs_pos: Whether to use absolute position embeddings.
        use_rel_pos: Whether to use relative position embeddings.
        rel_pos_bins: The number of relative position bins.
        max_rel_pos: The maximum relative position distance.
        spatial_embedding_dim: The size of the spatial embedding dimension.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=(112, 112),
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        spatial_embedding_dim=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.spatial_embedding_dim = spatial_embedding_dim
        
        # Input layers
        self.input_ids = layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        self.bbox = layers.Input(shape=(None, 4), dtype=tf.int32, name="bbox")
        self.attention_mask = layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
        self.image = layers.Input(shape=(*image_size, num_channels), dtype=tf.float32, name="image")
        
        # Embeddings
        self.word_embeddings = layers.Embedding(
            vocab_size, hidden_size, name="embeddings.word_embeddings"
        )
        self.position_embeddings = layers.Embedding(
            max_position_embeddings, hidden_size, name="embeddings.position_embeddings"
        )
        self.x_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.x_position_embeddings")
        self.y_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.y_position_embeddings")
        self.h_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.h_position_embeddings")
        self.w_position_embeddings = layers.Embedding(1024, spatial_embedding_dim, name="embeddings.w_position_embeddings")
        self.token_type_embeddings = layers.Embedding(
            type_vocab_size, hidden_size, name="embeddings.token_type_embeddings"
        )
        
        # Layer normalization
        self.embeddings_LayerNorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="embeddings.LayerNorm"
        )
        self.norm = layers.LayerNormalization(epsilon=layer_norm_eps, name="norm")
        
        # Spatial embedding projections
        self.x_proj = layers.Dense(hidden_size, name="x_proj")
        self.y_proj = layers.Dense(hidden_size, name="y_proj")
        self.h_proj = layers.Dense(hidden_size, name="h_proj")
        self.w_proj = layers.Dense(hidden_size, name="w_proj")
        
        # Transformer encoder layers
        self.encoder_layers = [
            LayoutLMv3TransformerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range,
                layer_norm_eps=layer_norm_eps,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                rel_pos_bins=rel_pos_bins,
                max_rel_pos=max_rel_pos,
                name=f"encoder.layer.{i}",
            )
            for i in range(num_hidden_layers)
        ]
        
        # Image processing
        self.patch_embed = layers.Conv2D(
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            name="patch_embed.proj",
        )
        self.patch_embed_layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="LayerNorm"
        )
        
        # CLS token
        self.cls_token = self.add_weight(
            shape=(1, 1, hidden_size),
            initializer="random_normal",
            trainable=True,
            name="cls_token",
        )
        
        # Pooler
        self.pooler = layers.Dense(hidden_size, activation="tanh", name="pooler")
        
    def call(self, inputs):
        input_ids = inputs["input_ids"]
        bbox = inputs["bbox"]
        attention_mask = inputs["attention_mask"]
        image = inputs["image"]
        
        # Get sequence length
        seq_length = tf.shape(input_ids)[1]
        
        # Create position IDs
        position_ids = tf.range(seq_length, dtype=tf.int32)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Get spatial embeddings
        x_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        y_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 2])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 3])
        
        # Project spatial embeddings to hidden size
        x_position_embeddings = self.x_proj(x_position_embeddings)
        y_position_embeddings = self.y_proj(y_position_embeddings)
        h_position_embeddings = self.h_proj(h_position_embeddings)
        w_position_embeddings = self.w_proj(w_position_embeddings)
        
        # Get word embeddings and token type embeddings
        word_embeddings = self.word_embeddings(input_ids)
        token_type_ids = tf.zeros_like(input_ids[:, 0:1])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        token_type_embeddings = tf.broadcast_to(
            token_type_embeddings,
            [tf.shape(input_ids)[0], tf.shape(input_ids)[1], self.hidden_size],
        )
        
        # Combine all embeddings
        text_embeddings = (
            word_embeddings
            + position_embeddings
            + x_position_embeddings
            + y_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        
        # Process image
        patch_embeddings = self.patch_embed(image)
        batch_size = tf.shape(patch_embeddings)[0]
        patch_embeddings_shape = tf.shape(patch_embeddings)
        num_patches = patch_embeddings_shape[1] * patch_embeddings_shape[2]
        patch_embeddings = tf.reshape(
            patch_embeddings, [batch_size, num_patches, self.hidden_size]
        )
        patch_embeddings = self.patch_embed_layer_norm(patch_embeddings)
        
        # Combine text and image embeddings
        x = tf.concat([text_embeddings, patch_embeddings], axis=1)
        
        # Add CLS token
        cls_tokens = tf.broadcast_to(
            self.cls_token, [tf.shape(x)[0], 1, self.hidden_size]
        )
        x = tf.concat([cls_tokens, x], axis=1)
        
        # Apply layer normalization
        x = self.embeddings_LayerNorm(x)
        
        # Create attention mask
        new_seq_length = tf.shape(x)[1]
        extended_attention_mask = tf.ones(
            (tf.shape(input_ids)[0], new_seq_length), dtype=tf.int32
        )
        extended_attention_mask = tf.cast(
            extended_attention_mask[:, tf.newaxis, tf.newaxis, :],
            dtype=tf.float32,
        )
        extended_attention_mask = tf.broadcast_to(
            extended_attention_mask,
            (tf.shape(input_ids)[0], self.num_attention_heads, new_seq_length, new_seq_length),
        )
        
        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x, extended_attention_mask)
        
        # Apply final layer normalization
        x = self.norm(x)
        
        # Apply pooler
        pooled_output = self.pooler(x[:, 0])
        
        return {
            "sequence_output": x,
            "pooled_output": pooled_output,
        }

@register_keras_serializable()
class LayoutLMv3TransformerLayer(layers.Layer):
    """Transformer layer for LayoutLMv3.
    
    Args:
        hidden_size: The size of the hidden layers.
        num_attention_heads: The number of attention heads.
        intermediate_size: The size of the intermediate layer.
        hidden_act: The activation function for the intermediate layer.
        hidden_dropout_prob: The dropout probability for the hidden layers.
        attention_probs_dropout_prob: The dropout probability for the attention probabilities.
        initializer_range: The standard deviation of the truncated normal initializer.
        layer_norm_eps: The epsilon value for layer normalization.
        qkv_bias: Whether to use bias in the query, key, value projections.
        use_rel_pos: Whether to use relative position embeddings.
        rel_pos_bins: The number of relative position bins.
        max_rel_pos: The maximum relative position distance.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_bins=32,
        max_rel_pos=128,
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
        
        # Attention layer
        self.attention = LayoutLMv3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=attention_probs_dropout_prob,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_bins=rel_pos_bins,
            max_rel_pos=max_rel_pos,
            name="attention",
        )
        
        # Layer normalization
        self.attention_output_dense = layers.Dense(hidden_size, name="attention.output.dense")
        self.attention_output_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="attention.output.LayerNorm"
        )
        
        # Intermediate layer
        self.intermediate_dense = layers.Dense(
            intermediate_size, activation=hidden_act, name="intermediate.dense"
        )
        
        # Output layer
        self.output_dense = layers.Dense(hidden_size, name="output.dense")
        self.output_layernorm = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="output.LayerNorm"
        )
        
        # Dropout
        self.dropout = layers.Dropout(hidden_dropout_prob)
        
    def call(self, hidden_states, attention_mask=None):
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_output_layernorm(attention_output + hidden_states)
        
        # Feed-forward
        intermediate_output = self.intermediate_dense(attention_output)
        intermediate_output = self.output_dense(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        output = self.output_layernorm(intermediate_output + attention_output)
        
        return output

@register_keras_serializable()
class LayoutLMv3Attention(layers.Layer):
    """Attention layer for LayoutLMv3.
    
    Args:
        hidden_size: The size of the hidden layers.
        num_attention_heads: The number of attention heads.
        dropout: The dropout probability.
        qkv_bias: Whether to use bias in the query, key, value projections.
        use_rel_pos: Whether to use relative position embeddings.
        rel_pos_bins: The number of relative position bins.
        max_rel_pos: The maximum relative position distance.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        dropout=0.1,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        
        # Query, key, value projections
        self.q_proj = layers.Dense(hidden_size, use_bias=qkv_bias, name="query")
        self.k_proj = layers.Dense(hidden_size, use_bias=qkv_bias, name="key")
        self.v_proj = layers.Dense(hidden_size, use_bias=qkv_bias, name="value")
        
        # Output projection
        self.out_proj = layers.Dense(hidden_size, name="output")
        
        # Dropout
        self.dropout_layer = layers.Dropout(dropout)
        
        # Relative position embeddings (if enabled)
        if use_rel_pos:
            self.rel_pos_bias = self.add_weight(
                shape=(2 * rel_pos_bins - 1, num_attention_heads),
                initializer="zeros",
                trainable=True,
                name="rel_pos_bias",
            )
    
    def call(self, hidden_states, attention_mask=None):
        batch_size = tf.shape(hidden_states)[0]
        seq_length = tf.shape(hidden_states)[1]
        
        # Project to query, key, value
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = tf.reshape(q, (batch_size, seq_length, self.num_attention_heads, -1))
        k = tf.reshape(k, (batch_size, seq_length, self.num_attention_heads, -1))
        v = tf.reshape(v, (batch_size, seq_length, self.num_attention_heads, -1))
        
        # Transpose for attention
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        # Compute attention scores
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        # Apply relative position bias if enabled
        if self.use_rel_pos:
            rel_pos_bias = self._get_rel_pos_bias(seq_length)
            attention_scores = attention_scores + rel_pos_bias
        
        # Apply softmax
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout_layer(attention_probs)
        
        # Apply attention to values
        context = tf.matmul(attention_probs, v)
        
        # Reshape and project output
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, seq_length, self.hidden_size))
        output = self.out_proj(context)
        
        return output
    
    def _get_rel_pos_bias(self, seq_length):
        """Get relative position bias."""
        # Create relative position indices
        pos = tf.range(seq_length)
        rel_pos = pos[:, None] - pos[None, :]
        rel_pos = rel_pos + self.rel_pos_bins - 1
        
        # Clip to valid range
        rel_pos = tf.clip_by_value(rel_pos, 0, 2 * self.rel_pos_bins - 2)
        
        # Get bias values
        bias = tf.gather(self.rel_pos_bias, rel_pos)
        
        # Reshape for attention
        bias = tf.transpose(bias, perm=[2, 0, 1])
        bias = tf.expand_dims(bias, 0)
        
        return bias 