import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.models.t5.t5_layer_norm import T5LayerNorm
from keras_hub.src.models.t5.t5_multi_head_attention import T5MultiHeadAttention


class T5TransformerLayer(keras.layers.Layer):
    """A single encoder or decoder layer of a T5 stack.

    Decoder layers add a cross-attention block over the encoder output, and
    support a static key/value cache for both attention blocks so that
    generation does not recompute the states of already seen tokens. See
    `T5MultiHeadAttention` for the caching modes.

    Args:
        is_decoder: bool. Whether this layer belongs to the decoder stack.
            Decoder layers add a cross-attention block and mask causally.
        hidden_dim: int. The size of the layer output.
        intermediate_dim: int. The output dimension of the first dense layer
            of the feedforward block.
        key_value_dim: int. The size of each attention head.
        dropout: float. Dropout probability for the layer.
        activation: string. The activation to use in the feedforward block.
        layer_norm_epsilon: float. Epsilon factor for the layer normalization
            layers.
        num_heads: int. The number of attention heads.
        use_gated_activation: bool. Whether to gate the inner dense block of
            the feedforward network.
        use_relative_attention_bias: bool. Whether the self-attention block
            owns the relative attention bias weights. Only the first layer of
            a stack should set this to `True`.

    Call arguments:
        position_bias: Optional relative attention bias returned by an earlier
            layer of the same stack. The first layer of a stack computes it,
            along with the causal mask, and the rest reuse it.
        use_causal_mask: bool. Whether to build a causal mask and combine it
            with `attention_mask`. The mask is rectangular during cached
            decoding, where the queries are a slice of the cached sequence.
        self_attention_cache: Optional key/value cache for the self-attention
            block, of shape
            `(batch_size, 2, target_length, num_heads, key_value_dim)`.
        self_attention_cache_update_index: Optional int or int tensor, the
            index at which to update `self_attention_cache`.
        cross_attention_cache: Optional key/value cache for the
            cross-attention block, of shape
            `(batch_size, 2, source_length, num_heads, key_value_dim)`.
        cross_attention_cache_update_index: Optional int or int tensor, the
            index at which to update `cross_attention_cache`. Pass `None` to
            reuse the cache without recomputing the encoder projections.

    Returns:
        A `(hidden_states, position_bias)` tuple, or a
        `(hidden_states, position_bias, self_attention_cache,
        cross_attention_cache)` tuple when `self_attention_cache` is set.
    """

    def __init__(
        self,
        is_decoder,
        hidden_dim,
        intermediate_dim,
        key_value_dim,
        dropout,
        activation,
        layer_norm_epsilon,
        num_heads,
        use_gated_activation=False,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_decoder = is_decoder
        self.use_gated_activation = use_gated_activation

        self.self_attention = T5MultiHeadAttention(
            is_decoder=is_decoder,
            hidden_dim=hidden_dim,
            key_value_dim=key_value_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_relative_attention_bias=use_relative_attention_bias,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.self_attention_layer_norm = T5LayerNorm(
            layer_norm_epsilon,
            dtype=self.dtype_policy,
        )
        self.self_attention_dropout = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

        if self.is_decoder:
            self.cross_attention = T5MultiHeadAttention(
                is_decoder=is_decoder,
                hidden_dim=hidden_dim,
                key_value_dim=key_value_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_relative_attention_bias=False,
                dtype=self.dtype_policy,
                name="cross_attention",
            )
            self.cross_attention_layer_norm = T5LayerNorm(
                layer_norm_epsilon,
                dtype=self.dtype_policy,
            )
            self.cross_attention_dropout = keras.layers.Dropout(
                dropout,
                dtype=self.dtype_policy,
            )

        self.input_projector = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            activation=keras.activations.get(activation),
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=hidden_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="input_projector",
        )
        if self.use_gated_activation:
            self.gate_projector = keras.layers.Dense(
                intermediate_dim,
                use_bias=False,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=hidden_dim**-0.5
                ),
                dtype=self.dtype_policy,
                name="gate_projector",
            )
        self.output_projector = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=intermediate_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="output_projector",
        )
        self.layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_causal_mask=False,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
        training=False,
    ):
        # Only the first layer of a stack builds the mask; the rest inherit it
        # through `position_bias`, which already has it folded in.
        if use_causal_mask and position_bias is None:
            attention_mask = self._compute_self_attention_mask(
                hidden_states,
                attention_mask,
                self_attention_cache,
                self_attention_cache_update_index,
            )

        x = hidden_states  # Intermediate result.

        residual = x
        x = self.self_attention_layer_norm(x)
        x, position_bias, self_attention_cache = self.self_attention(
            x,
            mask=attention_mask,
            position_bias=position_bias,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )
        x = self.self_attention_dropout(x, training=training)
        x = x + residual

        if self.is_decoder:
            residual = x
            x = self.cross_attention_layer_norm(x)
            x, _, cross_attention_cache = self.cross_attention(
                x,
                key_value_states=encoder_hidden_states,
                mask=encoder_attention_mask,
                cache=cross_attention_cache,
                cache_update_index=cross_attention_cache_update_index,
                training=training,
            )
            x = self.cross_attention_dropout(x, training=training)
            x = x + residual

        residual = x
        x = self.layer_norm(x)
        if self.use_gated_activation:
            hidden_activation = self.input_projector(x)
            hidden_linear = self.gate_projector(x)
            x = hidden_activation * hidden_linear
        else:
            x = self.input_projector(x)
        x = self.dropout_layer(x, training=training)
        x = self.output_projector(x)
        x = self.dropout_layer(x, training=training)
        x = x + residual

        # Keras rejects `None` inside a functional model's output structure, so
        # the caches are only returned when the caller actually passed some.
        if self_attention_cache is not None:
            return x, position_bias, self_attention_cache, cross_attention_cache
        return x, position_bias

    def _compute_self_attention_mask(
        self,
        hidden_states,
        attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        batch_size = ops.shape(hidden_states)[0]
        input_length = output_length = ops.shape(hidden_states)[1]
        # We need a rectangular causal mask when doing cached decoding. For
        # generative inference `hidden_states` will generally be length 1,
        # while the cache spans the full generation length.
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]
        causal_mask = compute_causal_mask(
            batch_size,
            input_length,
            output_length,
            (
                0
                if self_attention_cache_update_index is None
                else self_attention_cache_update_index
            ),
        )
        if attention_mask is None:
            return causal_mask
        return causal_mask & ops.cast(attention_mask, "bool")
