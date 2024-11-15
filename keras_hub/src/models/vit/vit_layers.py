import math

import keras
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


class TokenLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, input_shape[-1]),
            initializer="zeros",
            dtype=self.dtype_policy,
            name="cls_token",
        )
        self.built = True

    def call(self, inputs):
        cls_token = self.cls_token + keras.ops.zeros_like(inputs[:, 0:1])
        out = keras.ops.concatenate([cls_token, inputs], axis=1)

        return out


class MLP(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        mlp_dim,
        use_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === config ===
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense1 = keras.layers.Dense(
            units=self.mlp_dim,
            use_bias=self.use_bias,
            activation="gelu",
            bias_initializer=(
                keras.initializers.RandomNormal(stddev=1e-6)
                if self.use_bias
                else None
            ),
            dtype=self.dtype_policy,
            name="dense_1",
        )
        self.dense1.build(input_shape)
        self.dense2 = keras.layers.Dense(
            units=self.hidden_dim,
            use_bias=self.use_bias,
            bias_initializer=(
                keras.initializers.RandomNormal(stddev=1e-6)
                if self.use_bias
                else None
            ),
            dtype=self.dtype_policy,
            name="dense_2",
        )
        self.dense2.build((None, None, self.mlp_dim))
        self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")
        self.built = True

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        out = self.dropout(x)
        return out


class ViTPatchingAndEmbedding(keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_channels=3,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        num_patches = (image_size // patch_size) ** 2
        num_positions = num_patches + 1

        # === config ===
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_positions = num_positions
        self.data_format = standardize_data_format(data_format)

    def build(self, input_shape):
        self.patch_embedding = keras.layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            activation=None,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=math.sqrt(1 / (3 * self.patch_size * self.patch_size)),
            ),
            dtype=self.dtype_policy,
            data_format=self.data_format,
            name="patch_embedding",
        )
        self.patch_embedding.build(input_shape)
        self.token_layer = TokenLayer(dtype=self.dtype_policy)
        self.position_embedding = keras.layers.Embedding(
            self.num_positions,
            self.hidden_dim,
            dtype=self.dtype_policy,
            name="position_embedding",
        )
        self.position_embedding.build([1, self.num_positions])
        self.position_ids = ops.expand_dims(
            ops.arange(self.num_positions), axis=0
        )
        self.built = True

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        input_shape = ops.shape(x)  # (N, H, W, C) or (N, C, H, W)
        if self.data_format == "channels_first":
            x = ops.transpose(x, axes=(0, 2, 3, 1))
        x = ops.reshape(x, [input_shape[0], -1, input_shape[-1]])
        x = self.token_layer(x)
        x = x + self.position_embedding(self.position_ids)
        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.num_positions,
            self.hidden_dim,
        )


class ViTEncoderBlock(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate,
        attention_dropout,
        layer_norm_epsilon,
        **kwargs,
    ):
        super().__init__(**kwargs)

        key_dim = hidden_dim // num_heads

        # === config ===
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        # Attention block
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_1",
            dtype=self.dtype_policy,
        )
        self.layer_norm_1.build(input_shape)
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            use_bias=False,
            dropout=self.attention_dropout,
            name="mha",
            dtype=self.dtype_policy,
        )
        self.mha.build(input_shape, input_shape)
        self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")

        # MLP block
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_2",
            dtype=self.dtype_policy,
        )
        self.layer_norm_2.build((None, None, self.hidden_dim))
        self.mlp = MLP(
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            name="mlp",
            dtype=self.dtype_policy,
        )
        self.mlp((None, None, self.hidden_dim))
        self.built = True

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.mha(x, x)
        x = self.dropout(x)
        x = x + inputs

        y = self.layer_norm_2(x)
        y = self.mlp(y)

        return x + y


class ViTEncoder(keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === config ===
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        layers = []
        for i in range(self.num_layers):
            encoder_block = ViTEncoderBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                attention_dropout=self.attention_dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                name=f"tranformer_block_{i+1}",
            )
            encoder_block.build((None, None, self.hidden_dim))
            layers.append(encoder_block)

        self.encoder_layers = keras.Sequential(layers, name="encoder_layers")
        self.layer_norm = keras.layers.Normalization(
            self.layer_norm_epsilon, name="ln"
        )
        self.layer_norm.build((None, None, self.hidden_dim))

    def call(self, inputs):
        x = self.dropout(inputs)
        x = self.encoder_layers(x)
        x = self.layer_norm(x)
        return x
