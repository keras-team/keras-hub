import keras
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


class Dinov2PatchAndEmbeddings(keras.layers.Layer):
    """Patches the image and embeds the patches.

    Args:
        image_size: (int, int). Size of the input image.
        patch_size: (int, int). Size of each image patch.
        hidden_dim: int. Dimensionality of the patch embeddings.
        num_channels: int. Number of channels in the input image. Defaults to
            `3`.
        use_class_token: bool. Whether to use class token to be part of
            patch embedding. Defaults to `True`.
        data_format: str. `"channels_last"` or `"channels_first"`. Defaults to
            `None` (which uses `"channels_last"`).
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_channels=3,
        data_format=None,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        grid_size = tuple([s // p for s, p in zip(image_size, patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        num_positions = num_patches + 1

        # === Config ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_positions = num_positions
        self.dropout_rate = dropout_rate
        self.data_format = standardize_data_format(data_format)

    def build(self, input_shape):
        self.mask_token = self.add_weight(
            shape=(1, self.hidden_dim),
            initializer="zeros",
            dtype=self.variable_dtype,
            name="mask_token",
        )
        self.class_token = self.add_weight(
            shape=(
                1,
                1,
                self.hidden_dim,
            ),
            initializer="random_normal",
            dtype=self.variable_dtype,
            name="class_token",
        )
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
        self.position_embedding = keras.layers.Embedding(
            self.num_positions,
            self.hidden_dim,
            dtype=self.dtype_policy,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            name="position_embedding",
        )
        self.position_embedding.build((1, self.num_positions))
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        self.position_ids = keras.ops.expand_dims(
            keras.ops.arange(self.num_positions), axis=0
        )
        self.built = True

    def interpolate_pos_encoding(self, embeddings, height, width):
        """Interpolates positional embeddings for different image sizes."""
        num_patches = ops.shape(embeddings)[1] - 1
        num_positions = ops.shape(self.position_embedding)[1] - 1

        # If image size is unchanged, return as is
        if num_patches == num_positions and height == width:
            return self.position_embedding

        class_pos_embed = self.position_embedding[:, :1]  # CLS token position
        patch_pos_embed = self.position_embedding[:, 1:]  # Patch positions

        # Compute new patch grid size
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]
        patch_pos_embed = ops.reshape(
            patch_pos_embed,
            (1, int(num_positions**0.5), int(num_positions**0.5), -1),
        )

        # Interpolate the position embeddings
        patch_pos_embed = keras.layers.Resizing(
            new_height, new_width, interpolation="bicubic"
        )(patch_pos_embed)

        patch_pos_embed = ops.reshape(patch_pos_embed, (1, -1, self.hidden_dim))

        return ops.concatenate([class_pos_embed, patch_pos_embed], axis=1)

    def call(self, inputs, bool_masked_pos=None):
        patch_embeddings = self.patch_embedding(inputs)
        if self.data_format == "channels_first":
            patch_embeddings = ops.transpose(
                patch_embeddings, axes=(0, 2, 3, 1)
            )
        embeddings_shape = ops.shape(patch_embeddings)
        patch_embeddings = ops.reshape(
            patch_embeddings, [embeddings_shape[0], -1, embeddings_shape[-1]]
        )
        position_embeddings = self.position_embedding(self.position_ids)
        position_embeddings = self.interpolate_pos_encoding(
            position_embeddings, embeddings_shape[1], embeddings_shape[2]
        )

        class_token = ops.tile(self.class_token, (embeddings_shape[0], 1, 1))
        patch_embeddings = ops.concatenate(
            [class_token, patch_embeddings], axis=1
        )
        embeddings = ops.add(patch_embeddings, position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

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
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class DinoV2MLP(keras.layers.Layer):
    """Multi-Layer Perceptron (MLP) block.

    Args:
        hidden_dim: int. Dimensionality of the hidden representations.
        mlp_dim: int. Dimensionality of the intermediate MLP layer.
        use_bias: bool. Whether to use bias in the dense layers. Defaults to
            `True`.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to `0.0`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

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
        self.dense_1 = keras.layers.Dense(
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
        self.dense_1.build(input_shape)
        self.dense_2 = keras.layers.Dense(
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
        self.dense_2.build((None, None, self.mlp_dim))
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )
        self.built = True

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        out = self.dropout(x)
        return out


class DinoV2LayerScale(keras.layers.Layer):
    """LayerScale layer introduced in
    [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239v2).

    Args:
        init_value: int. Value to initialize the diagonal matrix of
            LayerScale.
        hidden_dim: int. Dimensionality of the hidden representations.
    """

    def __init__(self, init_value: float, hidden_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.lambda1 = self.add_weight(
            shape=(self.hidden_dim,),
            initializer=keras.initializers.Constant(self.init_value),
            dtype=self.dtype_policy,
        )
        self.built = True

    def call(self, x):
        return x * self.lambda1


class DinoV2DropPath(keras.layers.Layer):
    """Drop path (Stochastic Depth) per sample applied int path of residual
    blocks.

    """

    def __init__(self, drop_prob, seed=None):
        self.drop_prob = drop_prob
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=False):
        if self.drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - self.drop_prob
        input_shape = ops.shape(x)
        shape = (input_shape[0],) + (1,) * (len(input_shape) - 1)
        random_tensor = keep_prob + keras.random.normal(
            shape, dtype=self.dtype, seed=self.seed_generator
        )
        random_tensor = ops.floor(random_tensor)

        output = random_tensor / keep_prob * random_tensor
        return output


class DinoV2EncoderBlock(keras.layers.Layer):
    """DinoV2 encoder block.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. Dimensionality of the hidden representations.
        mlp_dim: int. Dimensionality of the intermediate MLP layer.
        use_mha_bias: bool. Whether to use bias in the multi-head attention
            layer. Defaults to `True`.
        use_mlp_bias: bool. Whether to use bias in the MLP layer. Defaults to
            `True`.
        drop_path_rate: float. Dropout rate. Between 0 and 1. Defaults to `0.0`.
        attention_dropout: float. Dropout rate for the attention mechanism.
            Between 0 and 1. Defaults to `0.0`.
        layer_norm_epsilon: float. Small float value for layer normalization
            stability. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        mlp_dim,
        use_mha_bias=True,
        use_mlp_bias=True,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        layer_scale_value=1.0,
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
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layer_scale_value = layer_scale_value

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
        self.drop_path = DinoV2DropPath(drop_prob=self.drop_path_rate)

        # LayerScale layers
        self.layer_scale_1 = DinoV2LayerScale(
            init_value=self.layer_scale_value,
            hidden_dim=self.hidden_dim,
            name="ls_1",
            dtype=self.dtype_policy,
        )
        self.layer_scale_1.build(input_shape)
        self.layer_scale_2 = DinoV2LayerScale(
            init_value=self.layer_scale_value,
            hidden_dim=self.hidden_dim,
            name="ls_2",
            dtype=self.dtype_policy,
        )
        self.layer_scale_2.build(input_shape)

        # MLP block
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name="ln_2",
            dtype=self.dtype_policy,
        )
        self.layer_norm_2.build((None, None, self.hidden_dim))
        self.mlp = DinoV2MLP(
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            use_bias=self.use_mlp_bias,
            dropout_rate=self.dropout_rate,
            name="mlp",
            dtype=self.dtype_policy,
        )
        self.mlp.build((None, None, self.hidden_dim))
        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        return_attention_scores=False,
    ):
        attention_scores = None
        x = self.layer_norm_1(hidden_states)
        if return_attention_scores:
            x, attention_scores = self.mha(
                x,
                x,
                attention_mask=attention_mask,
                return_attention_scores=return_attention_scores,
            )
        else:
            x = self.mha(
                x,
                x,
                attention_mask=attention_mask,
            )

        x = self.layer_scale_1(x)
        x = self.drop_path(x) + hidden_states

        y = self.layer_norm_2(x)
        y = self.mlp(y)
        y = self.layer_scale_2(x)
        y = self.drop_path(y)

        return x + y, attention_scores

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
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "layer_scale_value": self.layer_scale_value,
            }
        )
        return config


class DinoV2Encoder(keras.layers.Layer):
    """DinoV2 encoder.

    Args:
        num_layers: int. Number of Transformer encoder blocks.
        num_heads: int. Number of attention heads.
        hidden_dim: int. Dimensionality of the hidden representations.
        mlp_dim: int. Dimensionality of the intermediate MLP layer.
        use_mha_bias: bool. Whether to use bias in the multi-head attention
            layers. Defaults to `True`.
        use_mlp_bias: bool. Whether to use bias in the MLP layers. Defaults to
            `True`.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to `0.0`.
        attention_dropout: float. Dropout rate for the attention mechanism.
            Between 0 and 1. Defaults to `0.0`.
        layer_norm_epsilon: float. Small float value for layer normalization
            tability. Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        use_mha_bias=True,
        use_mlp_bias=True,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-6,
        layer_scale_value=1.0,
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
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.layer_scale_value = layer_scale_value

    def build(self, input_shape):
        self.encoder_layers = []
        for i in range(self.num_layers):
            encoder_block = DinoV2EncoderBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                drop_path_rate=self.drop_path_rate,
                use_mha_bias=self.use_mha_bias,
                use_mlp_bias=self.use_mlp_bias,
                attention_dropout=self.attention_dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"tranformer_block_{i + 1}",
            )
            encoder_block.build((None, None, self.hidden_dim))
            self.encoder_layers.append(encoder_block)
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="ln",
        )
        self.layer_norm.build((None, None, self.hidden_dim))
        self.built = True

    def call(
        self,
        hidden_states,
        attention_masks=None,
        output_hidden_states=False,
        return_attention_scores=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions_scores = () if return_attention_scores else None

        for i in range(self.num_layers):
            attention_mask = (
                attention_masks[i] if attention_masks is not None else None
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, scores = self.encoder_layers[i](
                hidden_states,
                attention_mask=attention_mask,
                return_attention_scores=return_attention_scores,
            )
            if return_attention_scores:
                all_self_attentions_scores = all_self_attentions_scores + (
                    scores,
                )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        x = self.layer_norm(hidden_states)
        return x, all_hidden_states, all_self_attentions_scores

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
                "drop_path_rate": self.drop_path_rate,
                "attention_dropout": self.attention_dropout,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "layer_scale_value": self.layer_scale_value,
            }
        )
        return config
