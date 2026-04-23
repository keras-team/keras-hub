"""BLIP-2 Q-Former model."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras.saving.register_keras_serializable(package="keras_hub")
class QFormerAttention(keras.layers.Layer):
    """Single attention block (self or cross) with post-LN residual."""

    def __init__(self, num_heads, hidden_dim, kv_dim, layer_norm_epsilon, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.kv_dim = kv_dim if kv_dim is not None else hidden_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.head_dim = hidden_dim // num_heads

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout, dtype=self.dtype_policy, name="dropout"
        )

    def build(self, inputs_shape):
        # inputs_shape is either a single shape or [query_shape, kv_shape]
        if isinstance(inputs_shape, (list, tuple)) and isinstance(inputs_shape[0], (list, tuple)):
            query_shape, kv_shape = inputs_shape[0], inputs_shape[1]
        else:
            query_shape = inputs_shape
            kv_shape = inputs_shape
        self.attention.build(query_shape, kv_shape)
        self.layer_norm.build(query_shape)
        super().build(inputs_shape)

    def call(self, inputs, training=None):
        # Accept either (query, key_value) list (functional graph)
        # or a single tensor for self-attention
        if isinstance(inputs, (list, tuple)):
            query, key_value = inputs[0], inputs[1]
        else:
            query, key_value = inputs, inputs

        attn = self.attention(query, key_value, key_value, training=training)
        attn = self.dropout_layer(attn, training=training)
        return self.layer_norm(query + attn)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "kv_dim": self.kv_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class QFormerLayer(keras.layers.Layer):
    """One Q-Former transformer block.

    In the functional graph this layer is called as:
        layer([query_tokens, vision_features])
    """

    def __init__(
        self,
        has_cross_attention,
        num_heads,
        hidden_dim,
        intermediate_dim,
        vision_dim,
        layer_norm_epsilon,
        dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.has_cross_attention = has_cross_attention
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.vision_dim = vision_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        self.self_attention = QFormerAttention(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            kv_dim=hidden_dim,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout=dropout,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        if has_cross_attention:
            self.cross_attention = QFormerAttention(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                kv_dim=vision_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                dtype=self.dtype_policy,
                name="cross_attention",
            )
        self.intermediate_dense = keras.layers.Dense(
            intermediate_dim,
            activation="gelu",
            use_bias=True,
            dtype=self.dtype_policy,
            name="intermediate_dense",
        )
        self.output_dense = keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="output_dense",
        )
        self.output_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="output_layer_norm",
        )
        self.output_dropout = keras.layers.Dropout(
            dropout, dtype=self.dtype_policy, name="output_dropout"
        )

    def build(self, inputs_shape):
        # When called from the functional graph with [query_tokens, vision_input],
        # Keras passes inputs_shape as a list of two shapes.
        if isinstance(inputs_shape, (list, tuple)) and isinstance(inputs_shape[0], (list, tuple)):
            query_shape = inputs_shape[0]
            vision_shape = inputs_shape[1] if len(inputs_shape) > 1 else None
        else:
            query_shape = inputs_shape
            vision_shape = None

        self.self_attention.build([query_shape, query_shape])
        if self.has_cross_attention and vision_shape is not None:
            self.cross_attention.build([query_shape, vision_shape])
        self.intermediate_dense.build(query_shape)
        ffn_out_shape = (query_shape[0], query_shape[1], self.intermediate_dim)
        self.output_dense.build(ffn_out_shape)
        self.output_layer_norm.build(query_shape)
        super().build(inputs_shape)

    def call(self, inputs, training=None):
        # Functional graph passes [query_tokens, vision_features]
        if isinstance(inputs, (list, tuple)):
            query_tokens, vision_features = inputs[0], inputs[1]
        else:
            query_tokens, vision_features = inputs, None

        x = self.self_attention([query_tokens, query_tokens], training=training)
        if self.has_cross_attention and vision_features is not None:
            x = self.cross_attention([x, vision_features], training=training)
        h = self.intermediate_dense(x)
        h = self.output_dense(h)
        h = self.output_dropout(h, training=training)
        return self.output_layer_norm(x + h)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "has_cross_attention": self.has_cross_attention,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "vision_dim": self.vision_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class BLIP2QueryTokens(keras.layers.Layer):
    """Holds the learned query-token weight and broadcasts over the batch.

    This thin wrapper exists so that the query_tokens parameter lives inside
    a proper Keras layer and participates cleanly in the functional graph.
    """

    def __init__(self, num_query_tokens, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_query_tokens = num_query_tokens
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.query_tokens = self.add_weight(
            shape=(1, self.num_query_tokens, self.hidden_dim),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
            name="query_tokens",
        )
        super().build(input_shape)

    def call(self, vision_features):
        # vision_features is only used to read the dynamic batch size
        batch_size = ops.shape(vision_features)[0]
        return ops.broadcast_to(
            self.query_tokens,
            (batch_size, self.num_query_tokens, self.hidden_dim),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_tokens": self.num_query_tokens,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


@keras_hub_export("keras_hub.models.BLIP2QFormer")
class BLIP2QFormer(keras.Model):
    """Querying Transformer (Q-Former) for BLIP-2.

    Functional Keras model — auto-builds on instantiation, so `.summary()`
    works without a manual `.build()` call.

    Args:
        num_query_tokens: int. Number of learnable query tokens.
        num_layers: int. Number of transformer layers.
        num_heads: int. Number of attention heads.
        hidden_dim: int. Transformer hidden size.
        intermediate_dim: int. FFN inner dimension.
        vision_dim: int. Dimension of input visual features.
        cross_attention_frequency: int. Insert cross-attention every N layers.
        dropout: float. Dropout probability.
        layer_norm_epsilon: float. LayerNorm epsilon.
    """

    def __init__(
        self,
        num_query_tokens,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        vision_dim,
        cross_attention_frequency,
        dropout,
        layer_norm_epsilon,
        name=None,
        **kwargs,
    ):
        query_tokens_layer = BLIP2QueryTokens(
            num_query_tokens=num_query_tokens,
            hidden_dim=hidden_dim,
            dtype=kwargs.get("dtype", None),
            name="query_tokens_layer",
        )
        layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=kwargs.get("dtype", None),
            name="layer_norm",
        )
        dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=kwargs.get("dtype", None),
            name="dropout",
        )
        transformer_layers = [
            QFormerLayer(
                has_cross_attention=(i % cross_attention_frequency == 0),
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                vision_dim=vision_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                dtype=kwargs.get("dtype", None),
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]

        vision_input = keras.Input(
            shape=(None, vision_dim), name="vision_features"
        )

        query_tokens = query_tokens_layer(vision_input)
        query_tokens = layer_norm(query_tokens)
        query_tokens = dropout_layer(query_tokens)

        for t_layer in transformer_layers:
            query_tokens = t_layer([query_tokens, vision_input])

        super().__init__(
            inputs=vision_input,
            outputs=query_tokens,
            name=name,
            **kwargs,
        )

        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.vision_dim = vision_dim
        self.cross_attention_frequency = cross_attention_frequency
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.query_tokens_layer = query_tokens_layer
        self.layer_norm = layer_norm
        self.transformer_layers = transformer_layers

    @property
    def query_tokens(self):
        return self.query_tokens_layer.query_tokens

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_tokens": self.num_query_tokens,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "vision_dim": self.vision_dim,
                "cross_attention_frequency": self.cross_attention_frequency,
                "dropout": self.dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config