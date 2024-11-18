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
            dtype=self.dtype,
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

        # === Config ===
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

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
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
            dtype=self.dtype_policy,
            data_format=self.data_format,
            name="patch_embedding",
        )
        self.patch_embedding.build(input_shape)
        self.token_layer = TokenLayer(dtype=self.dtype_policy)
        self.token_layer.build(input_shape)
        self.position_embedding = keras.layers.Embedding(
            self.num_positions,
            self.hidden_dim,
            dtype=self.dtype_policy,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            name="position_embedding",
        )
        self.position_embedding.build([1, self.num_positions])
        self.position_ids = self.add_weight(
            shape=(1, self.num_positions),
            initializer="zeros",
            # Let the backend determine the int dtype. For example, tf
            # requires int64 for correct device placement, whereas jax and torch
            # don't.
            dtype=int,
            trainable=False,
            name="position_ids",
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_channels": self.num_channels,
                "num_patches": self.num_patches,
                "num_positions": self.num_positions,
            }
        )
        return config


class ViTEncoderBlock(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        use_mha_bias=True,
        use_mlp_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        key_dim = hidden_dim // num_heads

        # === Config ===
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.mlp_dim = mlp_dim
        self.use_mha_bias = use_mha_bias
        self.use_mlp_bias = use_mlp_bias
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
            use_bias=self.use_mha_bias,
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
            use_bias=self.use_mlp_bias,
            name="mlp",
            dtype=self.dtype_policy,
        )
        self.mlp.build((None, None, self.hidden_dim))
        self.built = True

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.mha(x, x)
        x = self.dropout(x)
        x = x + inputs

        y = self.layer_norm_2(x)
        y = self.mlp(y)

        return x + y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "key_dim": self.key_dim,
                "mlp_dim": self.mlp_dim,
                "use_mha_bias": self.use_mha_bias,
                "use_mlp_bias": self.use_mlp_bias,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class ViTEncoder(keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        use_mha_bias=True,
        use_mlp_bias=True,
        dropout_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === config ===
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.use_mha_bias = use_mha_bias
        self.use_mlp_bias = use_mlp_bias
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        layers = []
        for i in range(self.num_layers):
            encoder_block = ViTEncoderBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                use_mha_bias=self.use_mha_bias,
                use_mlp_bias=self.use_mlp_bias,
                attention_dropout=self.attention_dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"tranformer_block_{i+1}",
            )
            encoder_block.build((None, None, self.hidden_dim))
            layers.append(encoder_block)
        self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")
        self.encoder_layers = keras.Sequential(layers, name="encoder_layers")
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="ln",
        )
        self.layer_norm.build((None, None, self.hidden_dim))
        self.built = True

    def call(self, inputs):
        x = self.dropout(inputs)
        x = self.encoder_layers(x)
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "mlp_dim": self.mlp_dim,
                "use_mha_bias": self.use_mha_bias,
                "use_mlp_bias": self.use_mlp_bias,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
