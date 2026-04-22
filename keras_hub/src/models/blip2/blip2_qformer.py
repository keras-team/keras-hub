"""BLIP-2 Q-Former model."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras.saving.register_keras_serializable(package="keras_hub")
class QFormerAttention(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        kv_dim=None,
        layer_norm_epsilon=1e-12,
        dropout=0.1,
        **kwargs,
    ):
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
            dropout,
            dtype=self.dtype_policy,
            name="dropout",
        )

    def build(self, input_shape, kv_shape=None):
        query_shape = input_shape
        if kv_shape is None:
            kv_shape = input_shape
        self.attention.build(query_shape, kv_shape)
        self.layer_norm.build(input_shape)
        super().build(input_shape)

    def call(self, query, key_value, training=None):
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
    def __init__(
        self,
        has_cross_attention,
        num_heads,
        hidden_dim,
        intermediate_dim,
        vision_dim=1408,
        layer_norm_epsilon=1e-12,
        dropout=0.1,
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
            dropout,
            dtype=self.dtype_policy,
            name="output_dropout",
        )

    def build(self, input_shape, vision_features_shape=None):
        self.self_attention.build(input_shape)
        if self.has_cross_attention:
            self.cross_attention.build(input_shape, vision_features_shape)
        self.intermediate_dense.build(input_shape)
        ffn_out_shape = (input_shape[0], input_shape[1], self.intermediate_dim)
        self.output_dense.build(ffn_out_shape)
        self.output_layer_norm.build(input_shape)
        super().build(input_shape)

    def call(self, query_tokens, vision_features=None, training=None):
        x = self.self_attention(query_tokens, query_tokens, training=training)
        if self.has_cross_attention and vision_features is not None:
            x = self.cross_attention(x, vision_features, training=training)
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


@keras_hub_export("keras_hub.models.Blip2QFormer")
class Blip2QFormer(keras.Model):
    """Querying Transformer (Q-Former) for BLIP-2.

    The Q-Former is a lightweight transformer that bridges the gap between a
    frozen vision encoder and a frozen language model. It uses a set of
    learnable query tokens to extract visual features from the vision encoder's
    output via cross-attention.

    The architecture consists of several transformer layers that alternate
    between self-attention (among query tokens) and cross-attention (between
    query tokens and visual features).

    References:
        - [Li et al., 2023](https://arxiv.org/abs/2301.12597)
        - [Salesforce LAVIS](https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models)

    Args:
        num_query_tokens: int. The number of learnable query tokens.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads.
        hidden_dim: int. The size of the transformer hidden state.
        intermediate_dim: int. The output dimension of the first Dense layer in
            the MLP block.
        vision_dim: int. The dimension of the input visual features.
        cross_attention_frequency: int. The frequency of cross-attention layers.
        dropout: float. Dropout probability for the transformer blocks.
        layer_norm_epsilon: float. The epsilon value for layer normalization.
        **kwargs: Standard Keras Model arguments.
    """

    def __init__(
        self,
        num_query_tokens,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        vision_dim=1408,
        cross_attention_frequency=2,
        dropout=0.1,
        layer_norm_epsilon=1e-12,
        name="qformer",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.vision_dim = vision_dim
        self.cross_attention_frequency = cross_attention_frequency
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.query_tokens = self.add_weight(
            shape=(1, num_query_tokens, hidden_dim),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
            name="query_tokens",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
            name="dropout",
        )
        self.transformer_layers = [
            QFormerLayer(
                has_cross_attention=(i % cross_attention_frequency == 0),
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                vision_dim=vision_dim,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout=dropout,
                dtype=self.dtype_policy,
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]

    def build(self, input_shape):
        query_shape = (None, self.num_query_tokens, self.hidden_dim)
        self.layer_norm.build(query_shape)
        for transformer_layer in self.transformer_layers:
            transformer_layer.build(query_shape, input_shape)
        super().build(input_shape)

    def call(self, vision_features, training=None):
        batch_size = ops.shape(vision_features)[0]
        query_tokens = ops.broadcast_to(
            self.query_tokens,
            (batch_size, self.num_query_tokens, self.hidden_dim),
        )
        query_tokens = self.layer_norm(query_tokens)
        query_tokens = self.dropout_layer(query_tokens, training=training)
        for transformer_layer in self.transformer_layers:
            query_tokens = transformer_layer(
                query_tokens, vision_features, training=training
            )
        return query_tokens

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
