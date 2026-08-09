import keras
from keras import ops

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)


class EdRecRMSNormalization(keras.layers.Layer):
    """RMSNorm layer that matches JAX EdRec implementation.

    Attributes:
        epsilon: float, epsilon value for numerical stability.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # JAX: rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        #                     + self.eps)
        # JAX: normed = x / rms
        # JAX: normed = normed * (1 + scale)

        # Standard RMSNorm is x * scale / rms.
        # EdRec RMSNorm is x * (1 + scale) / rms.
        # Note: If scale is initialized to ones, (1+scale) starts at 2.

        mean_square = ops.mean(ops.square(x), axis=-1, keepdims=True)
        rms = ops.sqrt(mean_square + self.epsilon)
        normed = x / rms
        return normed * ops.cast(1.0 + self.scale, x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class EdRecGatedFeedForward(keras.layers.Layer):
    """Gated FeedForward (GLU-style) layer.

    y = GELU(up_proj(x)) * gate_proj(x)
    y = down_proj(y)
    """

    def __init__(
        self,
        intermediate_dim,
        hidden_dim,
        dropout_rate=0.0,
        activation="gelu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim  # The output dimension (d_model)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.up_proj = keras.layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.gate_proj = keras.layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="gate_proj",
        )
        self.down_proj = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="down_proj",
        )
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )

    def call(self, x, training=False):
        # Up projection + activation (GELU)
        h = self.up_proj(x)
        if self.activation == "gelu":
            h = keras.activations.gelu(h, approximate=True)
        else:
            h = keras.activations.get(self.activation)(h)

        # Gate projection
        g = self.gate_proj(x)

        # Elementwise gating
        y = h * g

        # Down projection
        y = self.down_proj(y)

        # Dropout
        if self.dropout_rate > 0.0:
            y = self.dropout(y, training=training)

        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
            }
        )
        return config


class EdRecEncoderBlock(keras.layers.Layer):
    """EdRec Encoder Block.

    Pre-norm: x = x + Dropout(Attention(RMSNorm(x))) x = x +
    GatedFeedForward(RMSNorm(x))
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        dropout_rate=0.0,
        epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.head_dim = hidden_dim // num_heads

    def build(self, input_shape):
        self.pre_attention_norm = EdRecRMSNormalization(
            epsilon=self.epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_norm",
        )
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            output_shape=self.hidden_dim,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.dropout1 = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout1"
        )

        self.pre_ffw_norm = EdRecRMSNormalization(
            epsilon=self.epsilon, dtype=self.dtype_policy, name="pre_ffw_norm"
        )
        self.mlp = EdRecGatedFeedForward(
            intermediate_dim=self.intermediate_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="mlp",
        )

    def call(self, x, padding_mask=None, training=False):
        # Self Attention
        residual = x
        x_norm = self.pre_attention_norm(x)

        # padding_mask is [B, L]
        # We need to expand it to [B, 1, 1, L] for broadcasting against
        # [B, H, L, L]
        if padding_mask is not None:
            padding_mask = merge_padding_and_attention_mask(
                x, padding_mask, None
            )

        attn_out = self.attention(
            query=x_norm,
            value=x_norm,
            attention_mask=padding_mask,
            training=training,
        )
        attn_out = self.dropout1(attn_out, training=training)
        x = residual + attn_out

        # Feed Forward
        residual = x
        ff_norm = self.pre_ffw_norm(x)
        ff_out = self.mlp(ff_norm, training=training)
        x = residual + ff_out

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "dropout_rate": self.dropout_rate,
                "epsilon": self.epsilon,
            }
        )
        return config


class EdRecDecoderBlock(keras.layers.Layer):
    """EdRec Decoder Block.

    x = x + Dropout(SelfAttention(RMSNorm(x)))
    x = x + Dropout(CrossAttention(RMSNorm(x), encoder_outputs))
    x = x + GatedFeedForward(RMSNorm(x))
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        dropout_rate=0.0,
        epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.head_dim = hidden_dim // num_heads

    def build(self, input_shape):
        self.pre_self_attn_norm = EdRecRMSNormalization(
            epsilon=self.epsilon,
            dtype=self.dtype_policy,
            name="pre_self_attn_norm",
        )
        self.self_attention = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            output_shape=self.hidden_dim,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.dropout1 = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout1"
        )

        self.pre_cross_attn_norm = EdRecRMSNormalization(
            epsilon=self.epsilon,
            dtype=self.dtype_policy,
            name="pre_cross_attn_norm",
        )
        self.cross_attention = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            output_shape=self.hidden_dim,
            dtype=self.dtype_policy,
            name="cross_attention",
        )
        self.dropout2 = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout2"
        )

        self.pre_ffw_norm = EdRecRMSNormalization(
            epsilon=self.epsilon, dtype=self.dtype_policy, name="pre_ffw_norm"
        )
        self.mlp = EdRecGatedFeedForward(
            intermediate_dim=self.intermediate_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="mlp",
        )

    def call(
        self,
        x,
        encoder_outputs,
        decoder_padding_mask=None,
        encoder_padding_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
        use_causal_mask=True,
        training=False,
    ):
        # Self Attention
        residual = x
        x_norm = self.pre_self_attn_norm(x)

        batch_size = ops.shape(x)[0]
        input_length = ops.shape(x)[1]

        total_length = input_length
        if self_attention_cache is not None:
            total_length = ops.shape(self_attention_cache)[2]

        # Compute causal mask
        causal_mask = None
        if use_causal_mask:
            causal_mask = compute_causal_mask(
                batch_size,
                total_length,
                input_length,
                0
                if self_attention_cache_update_index is None
                else self_attention_cache_update_index,
            )

        # Merge with padding mask
        self_attn_mask = causal_mask
        if decoder_padding_mask is not None:
            # decoder_padding_mask is [B, L_dec]
            # merge_padding_and_attention_mask gives [B, 1, L, L]
            padding_mask_merged = merge_padding_and_attention_mask(
                x, decoder_padding_mask, None
            )

            if causal_mask is not None:
                self_attn_mask = ops.minimum(padding_mask_merged, causal_mask)
            else:
                self_attn_mask = padding_mask_merged

        self_attn_out = self.self_attention(
            query=x_norm,
            value=x_norm,
            attention_mask=self_attn_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )

        if self_attention_cache is not None:
            self_attn_out, self_attention_cache = self_attn_out

        self_attn_out = self.dropout1(self_attn_out, training=training)
        x = residual + self_attn_out

        # Cross Attention
        residual = x
        x_norm = self.pre_cross_attn_norm(x)

        cross_mask = None
        if encoder_padding_mask is not None:
            cross_mask = merge_padding_and_attention_mask(
                encoder_outputs, encoder_padding_mask, None
            )

        cross_attn_out = self.cross_attention(
            query=x_norm,
            value=encoder_outputs,
            attention_mask=cross_mask,
            cache=cross_attention_cache,
            cache_update_index=cross_attention_cache_update_index,
            training=training,
        )

        if cross_attention_cache is not None:
            cross_attn_out, cross_attention_cache = cross_attn_out

        cross_attn_out = self.dropout2(cross_attn_out, training=training)
        x = residual + cross_attn_out

        # Feed Forward
        residual = x
        ff_norm = self.pre_ffw_norm(x)
        ff_out = self.mlp(ff_norm, training=training)
        x = residual + ff_out

        if self_attention_cache is not None:
            if cross_attention_cache is not None:
                return x, self_attention_cache, cross_attention_cache
            return (
                x,
                self_attention_cache,
                ops.zeros((0,), dtype=self.compute_dtype),
            )
        return (
            x,
            ops.zeros((0,), dtype=self.compute_dtype),
            ops.zeros((0,), dtype=self.compute_dtype),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "dropout_rate": self.dropout_rate,
                "epsilon": self.epsilon,
            }
        )
        return config
