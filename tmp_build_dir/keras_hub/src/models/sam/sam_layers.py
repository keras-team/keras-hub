import math

import keras
from keras import ops


class MLP(keras.layers.Layer):
    """A MLP block with architecture.

    `input_dim -> [hidden_dim] * (num_layers - 1) -> output_dim`.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        output_dim: int. The number of units in the output layer.
        num_layers: int. The total number of dense layers to use.
        activation: str. Activation to use in the hidden layers.
            Default is `"relu"`.
    """

    def __init__(
        self, hidden_dim, output_dim, num_layers, activation="relu", **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        h = [hidden_dim] * (num_layers - 1)
        self.mlp_block = []
        for hidden_dim in h:
            self.mlp_block.append(
                keras.layers.Dense(hidden_dim, dtype=self.dtype_policy)
            )
            self.mlp_block.append(
                keras.layers.Activation(activation, dtype=self.dtype_policy)
            )
        self.mlp_block.append(
            keras.layers.Dense(output_dim, dtype=self.dtype_policy)
        )
        self.mlp_block = keras.models.Sequential(self.mlp_block)

    def build(self, input_shape):
        self.mlp_block.build(input_shape)
        self.built = True

    def call(self, x):
        return self.mlp_block(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
            }
        )
        return config


class MultiHeadAttentionWithDownsampling(keras.layers.Layer):
    """Multi-Head Attention with downsampling.

    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    This layer first downscales the features of input queries, keys, and
    values using a dense layer. Multi-head attention is then performed
    and the attention map is projected back (upscaled) to the number of
    input features.

    Args:
        num_heads: int. Number of attention heads.
        key_dim: int. Size of each attention head for query, key, and
            value.
        downsample_rate: int, optional. The factor by which to downscale the
            input features i.e. the input features of size `key_dim` are
            projected down to `key_dim // downsample_rate`.
    """

    def __init__(self, num_heads, key_dim, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.downsample_rate = downsample_rate
        self.internal_dims = key_dim // downsample_rate

        # Downsample
        self.query_proj = keras.layers.Dense(
            self.internal_dims * self.num_heads, dtype=self.dtype_policy
        )
        self.key_proj = keras.layers.Dense(
            self.internal_dims * self.num_heads, dtype=self.dtype_policy
        )
        self.value_proj = keras.layers.Dense(
            self.internal_dims * self.num_heads, dtype=self.dtype_policy
        )

        # Upsample
        self.out_proj = keras.layers.Dense(
            self.key_dim * self.num_heads, dtype=self.dtype_policy
        )

    def build(self, input_shape=None):
        self.query_proj.build([None, None, self.num_heads * self.key_dim])
        self.key_proj.build([None, None, self.num_heads * self.key_dim])
        self.value_proj.build([None, None, self.num_heads * self.key_dim])
        self.out_proj.build([None, None, self.internal_dims * self.num_heads])
        self.built = True

    def _separate_heads(self, x):
        shape = ops.shape(x)
        batch_size, N, channels = shape[0], shape[1], shape[2]
        x = ops.reshape(
            x, (batch_size, N, self.num_heads, channels // self.num_heads)
        )
        return ops.transpose(x, axes=(0, 2, 1, 3))

    def _recombine_heads(self, x):
        shape = ops.shape(x)
        batch_size, num_heads, N_T, channels_per_head = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        x = ops.transpose(x, axes=(0, 2, 1, 3))
        return ops.reshape(x, (batch_size, N_T, num_heads * channels_per_head))

    def call(self, query, value, key):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Separate into heads
        query = self._separate_heads(query)
        key = self._separate_heads(key)
        value = self._separate_heads(value)

        # Attention
        channels_per_head = ops.shape(query)[-1]
        out = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        out = out / ops.sqrt(
            ops.cast(channels_per_head, dtype=self.compute_dtype)
        )
        out = ops.softmax(out, axis=-1)

        # Get output
        attention_map = out @ value
        attention_map = self._recombine_heads(attention_map)
        return self.out_proj(attention_map)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config


class TwoWayMultiHeadAttention(keras.layers.Layer):
    """Two-way multi-head attention layer.

    Args:
        num_heads: int. Number of attention heads.
        key_dim: int. Size of each attention head for query, key, and
            value.
        intermediate_dim: int. Number of hidden dims to use in the mlp block.
        skip_first_layer_pos_embedding: bool. A boolean indicating whether to
            skip the first layer positional embeddings.
        attention_downsample_rate: int, optional. The downsample rate to use
            in the attention layers. Defaults to 2.
        activation: str, optional. The activation for the mlp block's output
            layer. Defaults to "relu".
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        intermediate_dim,
        skip_first_layer_pos_embedding,
        attention_downsample_rate=2,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.intermediate_dim = intermediate_dim
        self.skip_first_layer_pos_embedding = skip_first_layer_pos_embedding
        self.attention_downsample_rate = attention_downsample_rate
        self.activation = activation

        self.self_attention = MultiHeadAttentionWithDownsampling(
            num_heads=num_heads, key_dim=key_dim, dtype=self.dtype_policy
        )
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )
        self.cross_attention_token_to_image = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=key_dim,
                downsample_rate=attention_downsample_rate,
                dtype=self.dtype_policy,
            )
        )
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

        self.mlp_block = MLP(
            intermediate_dim,
            key_dim * num_heads,
            num_layers=2,
            activation=activation,
            dtype=self.dtype_policy,
        )

        self.layer_norm3 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )
        self.cross_attention_image_to_token = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=key_dim,
                downsample_rate=attention_downsample_rate,
                dtype=self.dtype_policy,
            )
        )
        self.layer_norm4 = keras.layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy
        )

    def build(self, input_shape=None):
        self.self_attention.build()
        self.layer_norm1.build([None, None, self.num_heads * self.key_dim])
        self.cross_attention_token_to_image.build()
        self.layer_norm2.build([None, None, self.num_heads * self.key_dim])
        self.mlp_block.build([None, None, self.num_heads * self.key_dim])
        self.layer_norm3.build([None, None, self.num_heads * self.key_dim])
        self.cross_attention_image_to_token.build()
        self.layer_norm4.build([None, None, self.num_heads * self.key_dim])
        self.built = True

    def call(self, queries, keys, query_pos_embedding, key_pos_embedding):
        if self.skip_first_layer_pos_embedding:
            queries = self.self_attention(
                query=queries, value=queries, key=queries
            )
        else:
            queries_with_pos_embedding = queries + query_pos_embedding
            attention_map = self.self_attention(
                query=queries_with_pos_embedding,
                key=queries_with_pos_embedding,
                value=queries,
            )
            queries = queries + attention_map
        queries = self.layer_norm1(queries)

        queries_with_pos_embedding = queries + query_pos_embedding
        keys_with_pos_embedding = keys + key_pos_embedding
        attention_map = self.cross_attention_token_to_image(
            query=queries_with_pos_embedding,
            key=keys_with_pos_embedding,
            value=keys,
        )
        queries = queries + attention_map
        queries = self.layer_norm2(queries)

        mlp_out = self.mlp_block(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        queries_with_pos_embedding = queries + query_pos_embedding
        keys_with_pos_embedding = keys + key_pos_embedding
        attention_map = self.cross_attention_image_to_token(
            query=keys_with_pos_embedding,
            key=queries_with_pos_embedding,
            value=queries,
        )
        keys = keys + attention_map
        keys = self.layer_norm4(keys)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "intermediate_dim": self.intermediate_dim,
                "skip_first_layer_pos_embedding": (
                    self.skip_first_layer_pos_embedding
                ),
                "attention_downsample_rate": self.attention_downsample_rate,
                "activation": self.activation,
            }
        )
        return config


class RandomFrequencyPositionalEmbeddings(keras.layers.Layer):
    """Positional encoding using random spatial frequencies.

    This layer maps coordinates/points in 2D space to positional
    encodings using random spatial frequencies.

    Args:
        num_positional_features: int. Number of positional features
            in the output.
        scale: float. The standard deviation of the random frequencies.
    """

    def __init__(self, num_positional_features, scale, **kwargs):
        super().__init__(**kwargs)
        self.num_positional_features = num_positional_features
        self.scale = scale
        self.positional_encoding_gaussian_matrix = self.add_weight(
            name="positional_encoding_gaussian_matrix",
            shape=(2, self.num_positional_features),
            dtype=self.variable_dtype,
            trainable=False,
            initializer=keras.initializers.get("normal"),
        )

    def build(self, input_shape=None):
        self.built = True

    def _positional_encodings(self, coords):
        coords = coords * 2 - 1
        coords = coords @ ops.cast(
            self.positional_encoding_gaussian_matrix, dtype=self.compute_dtype
        )
        coords = coords * (2 * math.pi)
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def call(self, size):
        return self.encode_image(size)

    def encode_image(self, size):
        """Generate a positional encoding for an image of any given size.
        Args:
            size: tuple[int, int]. The size of the image.
        Returns:
            tensor: Positional encoding of the image.
        """
        height, width = size
        grid = ops.ones(shape=(height, width), dtype=self.compute_dtype)
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / ops.cast(height, self.compute_dtype)
        x_embed = x_embed / ops.cast(width, self.compute_dtype)
        return self._positional_encodings(
            ops.stack([x_embed, y_embed], axis=-1)
        )

    def encode_coordinates(self, coords_input, image_size):
        """Positionally encode points that are not normalized to `[0, 1]`.
        Args:
            coords_input: tensor. 2D coordinates/points to map.
            image_size: tuple[int, int]. Height and width of the image
                being prompted.
        Returns:
            tensor: Positional encodings of the normalized coordinates.
        """
        coords_normalized = ops.stack(
            [
                coords_input[..., 0] / image_size[1],
                coords_input[..., 1] / image_size[0],
            ],
            axis=-1,
        )
        return self._positional_encodings(coords_normalized)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_positional_features": self.num_positional_features,
                "scale": self.scale,
            }
        )
        return config
