import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.utils.keras_utils import standardize_data_format


class VideoPrismFactorizedReshape(keras.layers.Layer):
    """Reshape layer for VideoPrism video inputs.

    Args:
        image_shape: tuple. The shape of a single image frame
            (height, width, channels) or (channels, height, width).
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
    """

    def __init__(self, image_shape, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.image_shape = image_shape
        self.data_format = standardize_data_format(data_format)

    def call(self, inputs):
        if self.data_format == "channels_first":
            inputs = ops.transpose(inputs, (0, 1, 3, 4, 2))
            channels, height, width = self.image_shape
        else:
            height, width, channels = self.image_shape
        return ops.reshape(inputs, (-1, height, width, channels))

    def get_config(self):
        config = super().get_config()
        config.update({"image_shape": self.image_shape})
        return config

    def compute_output_shape(self, inputs_shape):
        output_dim0 = None
        if inputs_shape[0] is not None and inputs_shape[1] is not None:
            output_dim0 = inputs_shape[0] * inputs_shape[1]
        if self.data_format == "channels_first":
            channels, height, width = self.image_shape
        else:
            height, width, channels = self.image_shape
        output_shape = (output_dim0, height, width, channels)
        return output_shape


class VideoPrismFactorizedDecoding(keras.layers.Layer):
    def __init__(self, num_patches, num_frames, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

    def call(self, inputs):
        outputs = ops.reshape(
            inputs,
            (-1, self.num_patches, self.num_frames, self.hidden_dim),
        )
        return ops.transpose(outputs, (0, 2, 1, 3))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_patches": self.num_patches,
                "num_frames": self.num_frames,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return [None, self.num_frames, self.num_patches, self.hidden_dim]


class VideoPrismMLP(keras.layers.Layer):
    """Multi-Layer Perceptron (MLP) block for VideoPrism.

    Args:
        hidden_dim: int. Dimensionality of the hidden representations.
        intermediate_dim: int. Dimensionality of the intermediate MLP layer.
        use_bias: bool. Whether to use bias in the dense layers. Defaults to
            `True`.
        dropout_rate: float. Dropout rate. Between 0 and 1. Defaults to `0.0`.
        activation: str. Activation function to use. Defaults to `"gelu"`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        use_bias=True,
        dropout_rate=0.0,
        activation="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # === Config ===
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.dense_1 = keras.layers.Dense(
            units=self.intermediate_dim,
            use_bias=self.use_bias,
            activation=activation,
            bias_initializer=(
                keras.initializers.RandomNormal(stddev=1e-6)
                if self.use_bias
                else None
            ),
            dtype=self.dtype_policy,
            name="dense_1",
        )
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
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )

    def build(self, inputs_shape):
        self.dense_1.build(inputs_shape)
        inputs_shape = self.dense_1.compute_output_shape(inputs_shape)
        self.dense_2.build(inputs_shape)
        inputs_shape = self.dense_2.compute_output_shape(inputs_shape)
        self.dropout.build(inputs_shape)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dropout(x)
        x = self.dense_2(x)
        out = self.dropout(x)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "use_bias": self.use_bias,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        output_shape = list(inputs_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape


class VideoPrismPerDimScale(keras.layers.Layer):
    """A layer to scale individual dimensions of the input.

    It returns `per_dim_scale * inputs / sqrt(dim)`.
    """

    def build(self, input_shape):
        self.dim = int(input_shape[-1])
        self.per_dim_scale = self.add_weight(
            shape=(self.dim,),
            initializer="zeros",
            name="per_dim_scale",
        )

    def call(self, inputs):
        r_softplus_0 = 1.442695041  # 1.0/softplus(0.0) = 1.442695041
        scale = ops.cast(
            r_softplus_0 / ops.sqrt(ops.cast(self.dim, "float32")),
            self.compute_dtype,
        )
        scale = ops.multiply(scale, ops.softplus(self.per_dim_scale))
        return ops.multiply(inputs, scale)

    def compute_output_shape(self, inputs_shape):
        return inputs_shape


class VideoPrismAttention(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        hidden_dim,
        dropout_rate=0,
        attention_logit_soft_cap=None,
        use_per_dim_scale=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.use_per_dim_scale = use_per_dim_scale

        self.query_dense = keras.layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_heads, self.key_dim),
            bias_axes="nh",
            dtype=self.dtype_policy,
            name="query",
        )
        self.key_dense = keras.layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_heads, self.key_dim),
            bias_axes="nh",
            dtype=self.dtype_policy,
            name="key",
        )
        self.value_dense = keras.layers.EinsumDense(
            equation="btd,dnh->btnh",
            output_shape=(None, self.num_heads, self.key_dim),
            bias_axes="nh",
            dtype=self.dtype_policy,
            name="value",
        )
        self.output_dense = keras.layers.EinsumDense(
            equation="btnh,nhd->btd",
            output_shape=(None, self.hidden_dim),
            bias_axes="d",
            dtype=self.dtype_policy,
            name="output",
        )
        self.dropout_layer = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )
        if self.use_per_dim_scale:
            self.per_dim_scale_layer = VideoPrismPerDimScale(
                dtype=self.dtype_policy, name="per_dim_scale"
            )

    def build(self, input_shape):
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        self.output_dense.build(
            list(input_shape)[:-1] + [self.num_heads, self.key_dim]
        )
        if self.use_per_dim_scale:
            self.per_dim_scale_layer.build(
                (None, None, self.num_heads, self.key_dim)
            )

    def call(self, query, value, key=None, attention_mask=None, training=None):
        if key is None:
            key = query

        q = self.query_dense(query)
        k = self.key_dense(key)
        v = self.value_dense(value)

        if self.use_per_dim_scale:
            q = self.per_dim_scale_layer(q)
        else:
            q = ops.multiply(
                q, ops.rsqrt(ops.cast(self.key_dim, self.compute_dtype))
            )

        score = ops.einsum("btnh,bsnh->bnts", q, k)
        if self.attention_logit_soft_cap is not None:
            score = ops.multiply(
                self.attention_logit_soft_cap,
                ops.tanh(ops.divide(score, self.attention_logit_soft_cap)),
            )

        # Use float32 for softmax stability
        score = ops.cast(score, "float32")
        if attention_mask is not None:
            mask = attention_mask
            if len(mask.shape) == 3:
                mask = ops.expand_dims(mask, axis=1)
            elif len(mask.shape) == 2:
                mask = ops.expand_dims(ops.expand_dims(mask, axis=1), axis=1)
            score = ops.subtract(
                score, ops.multiply(1.0 - ops.cast(mask, "float32"), 1e9)
            )
        weights = ops.softmax(score, axis=-1)
        weights = ops.cast(weights, self.compute_dtype)
        weights = self.dropout_layer(weights, training=training)

        context = ops.einsum("bnts,bsnh->btnh", weights, v)
        output = self.output_dense(context)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "use_per_dim_scale": self.use_per_dim_scale,
            }
        )
        return config

    def compute_output_shape(
        self,
        query_shape,
        value_shape,
        key_shape=None,
        attention_mask_shape=None,
    ):
        return query_shape


class VideoPrismEncoderBlock(keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        attention_logit_soft_cap=None,
        activation="gelu",
        is_causal=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.key_dim = hidden_dim // num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.activation = activation
        self.is_causal = is_causal

        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="ln_1",
        )
        self.mha = VideoPrismAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.attention_dropout_rate,
            attention_logit_soft_cap=self.attention_logit_soft_cap,
            dtype=self.dtype_policy,
            name="mha",
        )
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="ln_2",
        )
        self.mlp = VideoPrismMLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            use_bias=True,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            dtype=self.dtype_policy,
            name="mlp",
        )

    def build(
        self, inputs_shape, padding_mask_shape=None, attention_mask_shape=None
    ):
        self.layer_norm_1.build(inputs_shape)
        self.mha.build(inputs_shape)
        self.dropout.build(inputs_shape)
        self.layer_norm_2.build(inputs_shape)
        self.mlp.build(inputs_shape)

    def _compute_attention_mask(self, x, padding_mask):
        if padding_mask is None and not self.is_causal:
            return None

        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        if padding_mask is not None:
            attention_mask = ops.cast(padding_mask, "int32")
            attention_mask = ops.expand_dims(attention_mask, axis=1)
        else:
            attention_mask = ops.ones((batch_size, 1, seq_len), dtype="int32")

        if self.is_causal:
            causal_mask = compute_causal_mask(batch_size, seq_len, seq_len)
            attention_mask = ops.minimum(attention_mask, causal_mask)

        return attention_mask

    def call(self, inputs, padding_mask=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = self._compute_attention_mask(inputs, padding_mask)

        x = self.layer_norm_1(inputs)
        x = self.mha(x, x, attention_mask=attention_mask)
        x = self.dropout(x)
        x = ops.add(x, inputs)
        y = self.layer_norm_2(x)
        y = self.mlp(y)
        return ops.add(x, y)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "key_dim": self.key_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "activation": self.activation,
                "is_causal": self.is_causal,
            }
        )
        return config

    def compute_output_shape(
        self, inputs_shape, padding_mask_shape=None, attention_mask_shape=None
    ):
        output_shape = list(inputs_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape


class VideoPrismEncoder(keras.layers.Layer):
    """VideoPrism encoder."""

    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        attention_logit_soft_cap=None,
        activation="gelu",
        is_causal=False,
        use_final_layernorm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_logit_soft_cap = attention_logit_soft_cap
        self.activation = activation
        self.is_causal = is_causal
        self.use_final_layernorm = use_final_layernorm

        self.encoder_layers = []
        for i in range(self.num_layers):
            encoder_block = VideoPrismEncoderBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                layer_norm_epsilon=self.layer_norm_epsilon,
                attention_logit_soft_cap=self.attention_logit_soft_cap,
                activation=self.activation,
                is_causal=self.is_causal,
                dtype=self.dtype_policy,
                name=f"transformer_block_{i}",
            )
            self.encoder_layers.append(encoder_block)
        self.dropout = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )
        if self.use_final_layernorm:
            self.layer_norm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="ln",
            )
        else:
            self.layer_norm = None

    def build(
        self, inputs_shape, padding_mask_shape=None, attention_mask_shape=None
    ):
        self.dropout.build(inputs_shape)
        for i in range(self.num_layers):
            self.encoder_layers[i].build(inputs_shape)
            inputs_shape = self.encoder_layers[i].compute_output_shape(
                inputs_shape
            )
        if self.use_final_layernorm:
            self.layer_norm.build(inputs_shape)

    def call(self, inputs, padding_mask=None, attention_mask=None):
        x = self.dropout(inputs)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](
                x, padding_mask=padding_mask, attention_mask=attention_mask
            )
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout_rate": self.attention_dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_logit_soft_cap": self.attention_logit_soft_cap,
                "activation": self.activation,
                "is_causal": self.is_causal,
                "use_final_layernorm": self.use_final_layernorm,
            }
        )
        return config

    def compute_output_shape(
        self, inputs_shape, padding_mask_shape=None, attention_mask_shape=None
    ):
        output_shape = list(inputs_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape


class SinusoidalPositionalEmbedding(keras.layers.Layer):
    def __init__(
        self, hidden_dim, min_timescale=1.0, max_timescale=1.0e4, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.num_timescales = hidden_dim // 2

    def call(self, inputs):
        seq_length = ops.shape(inputs)[1]
        position = ops.cast(ops.arange(seq_length), "float32")[None, :]

        log_timescale_increment = ops.divide(
            ops.log(float(self.max_timescale) / float(self.min_timescale)),
            ops.maximum(ops.cast(self.num_timescales, "float32") - 1, 1.0),
        )
        inv_timescales = ops.multiply(
            self.min_timescale,
            ops.exp(
                ops.multiply(
                    ops.cast(ops.arange(self.num_timescales), "float32"),
                    -log_timescale_increment,
                )
            ),
        )
        scaled_time = ops.multiply(
            ops.expand_dims(position, 2),
            ops.expand_dims(inv_timescales, [0, 1]),
        )
        signal = ops.concatenate(
            [ops.sin(scaled_time), ops.cos(scaled_time)], axis=-1
        )

        # Pad if hidden_dim is odd
        if self.hidden_dim % 2 == 1:
            signal = ops.pad(signal, [[0, 0], [0, 0], [0, 1]])
        return ops.cast(signal, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "min_timescale": self.min_timescale,
                "max_timescale": self.max_timescale,
            }
        )
        return config


class VideoPrismClassToken(keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, inputs_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=keras.initializers.RandomNormal(
                stddev=1.0 / (self.hidden_dim**0.5)
            ),
            name="cls_token",
        )

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        cls_token = ops.multiply(
            self.cls_token,
            ops.sqrt(ops.cast(self.hidden_dim, self.compute_dtype)),
        )
        return ops.tile(cls_token, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config

    def compute_output_shape(self, inputs_shape):
        return (inputs_shape[0], 1, self.hidden_dim)


class VideoPrismEmbedding(keras.layers.Layer):
    """VideoPrism embedding layer.

    This layer combines token embedding, positional embedding, and class token
    addition. It also constructs the attention mask.

    Args:
        vocabulary_size: int. The size of the vocabulary.
        hidden_dim: int. The dimensionality of the hidden representations.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(self, vocabulary_size, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim

        self.token_embedding = keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="token_embedding",
        )
        self.position_embedding = SinusoidalPositionalEmbedding(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="position_embedding",
        )
        self.class_token = VideoPrismClassToken(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="class_token",
        )

    def build(self, token_ids_shape, padding_mask_shape=None):
        self.token_embedding.build(token_ids_shape)
        self.position_embedding.build(token_ids_shape)

    def call(self, token_ids, padding_mask=None):
        # Token Embedding
        x = self.token_embedding(token_ids)
        x = ops.multiply(
            x, ops.sqrt(ops.cast(self.hidden_dim, self.compute_dtype))
        )

        # Positional Embedding
        pos = self.position_embedding(token_ids)
        x = ops.add(x, pos)

        # Class Token
        cls = self.class_token(x)
        x = ops.concatenate([x, cls], axis=1)

        if padding_mask is not None:
            class_token_mask = ops.ones_like(padding_mask[:, :1])
            full_padding_mask = ops.concatenate(
                [padding_mask, class_token_mask], axis=1
            )
        else:
            full_padding_mask = None

        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        causal_mask = compute_causal_mask(batch_size, seq_len, seq_len)
        attention_mask = merge_padding_and_attention_mask(
            x, full_padding_mask, causal_mask
        )
        return x, attention_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


class VideoPrismPatchingAndEmbedding(keras.layers.Layer):
    """Patches the image and embeds the patches.

    This layer is a replacement for `ViTPatchingAndEmbedding` that strictly
    follows the Flax implementation of VideoPrism, using a Dense layer
    instead of Conv2D for patch projection.

    Args:
        image_size: (int, int). Size of the input image.
        patch_size: (int, int). Size of each image patch.
        hidden_dim: int. Dimensionality of the patch embeddings.
        num_channels: int. Number of channels in the input image. Defaults to
            `3`.
        dtype: The dtype of the layer weights.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_dim,
        num_channels=3,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels

        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patches_flattened_dim = (
            patch_size[0] * patch_size[1] * num_channels
        )

        self.patch_embedding = keras.layers.Dense(
            units=self.hidden_dim,
            dtype=self.dtype_policy,
            name="patch_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="position_embedding",
        )

    def build(self, inputs_shape):
        self.patch_embedding.build((None, None, self.patches_flattened_dim))
        self.position_embedding.build((1, self.num_patches))
        self.position_ids = ops.expand_dims(
            ops.arange(self.num_patches, dtype="int32"), axis=0
        )

    def call(self, inputs):
        # Reshape to (B, grid_h, patch_h, grid_w, patch_w, C)
        x = ops.reshape(
            inputs,
            (
                -1,
                self.grid_size[0],
                self.patch_size[0],
                self.grid_size[1],
                self.patch_size[1],
                self.num_channels,
            ),
        )
        # Transpose to (B, grid_h, grid_w, patch_h, patch_w, C)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = ops.reshape(x, (-1, self.num_patches, self.patches_flattened_dim))
        patch_embeddings = self.patch_embedding(x)
        pos_embeddings = self.position_embedding(self.position_ids)
        return ops.add(patch_embeddings, pos_embeddings)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_dim": self.hidden_dim,
                "num_channels": self.num_channels,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return (None, self.num_patches, self.hidden_dim)


class VideoPrismTemporalEmbedding(keras.layers.Layer):
    """VideoPrism temporal embedding layer.

    This layer handles the reshaping and temporal position embedding addition
    for the temporal encoder.

    Args:
        num_frames: int. The number of frames in the input video.
        hidden_dim: int. The dimensionality of the hidden representations.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`
    """

    def __init__(self, num_frames, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        self.embedding = keras.layers.Embedding(
            input_dim=self.num_frames,
            output_dim=self.hidden_dim,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="temporal_position_embedding",
        )

    def build(self, input_shape):
        self.embedding.build((1, self.num_frames))
        self.pos_ids = ops.expand_dims(
            ops.arange(self.num_frames, dtype="int32"), axis=0
        )

    def call(self, inputs):
        # Input shape: (B*T, Np, D) from spatial encoder
        num_patches = ops.shape(inputs)[1]
        x = ops.reshape(
            inputs, (-1, self.num_frames, num_patches, self.hidden_dim)
        )
        # Permute to (B, Np, T, D)
        x = ops.transpose(x, (0, 2, 1, 3))
        # Reshape to (B*Np, T, D)
        x = ops.reshape(x, (-1, self.num_frames, self.hidden_dim))

        # Add position embedding
        pos_emb = self.embedding(self.pos_ids)
        return ops.add(x, ops.cast(pos_emb, x.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_frames": self.num_frames,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_dim0 = None
        if input_shape[0] is not None and input_shape[1] is not None:
            output_dim0 = (input_shape[0] // self.num_frames) * input_shape[1]
        return (output_dim0, self.num_frames, self.hidden_dim)


class VideoPrismAttenTokenPoolingLayer(keras.layers.Layer):
    """Attentional token pooling layer.

    Args:
        num_heads: int. Number of attention heads.
        hidden_dim: int. The dimensionality of the hidden representations.
        query_dim: int. The dimensionality of the query vectors.
            Defaults to `hidden_dim` if not specified.
        num_queries: int. Number of attention queries. Defaults to `1`.
        dropout_rate: float. Dropout rate. Defaults to `0.0`.
        layer_norm_epsilon: float. Epsilon for layer normalization.
            Defaults to `1e-6`.
        **kwargs: Additional keyword arguments passed to `keras.layers.Layer`.
    """

    def __init__(
        self,
        num_heads,
        hidden_dim,
        query_dim=None,
        num_queries=1,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        use_per_dim_scale=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim or hidden_dim
        self.num_queries = num_queries
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_per_dim_scale = use_per_dim_scale
        key_dim = 4 * self.hidden_dim // self.num_heads

        self.pooling_attention_query = self.add_weight(
            shape=(self.num_queries, self.query_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            name="pooling_attention_query",
        )

        self.pooling_attention = VideoPrismAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            use_per_dim_scale=self.use_per_dim_scale,
            dtype=self.dtype_policy,
            name="pooling_attention",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pooling_attention_layer_norm",
        )
        self.dropout_layer = keras.layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="attention_dropout"
        )

    def build(self, inputs_shape, padding_mask_shape=None):
        query_shape = (inputs_shape[0], self.num_queries, self.query_dim)
        self.pooling_attention.build(query_shape)
        self.layer_norm.build(query_shape)
        self.dropout_layer.build(query_shape)

    def call(self, inputs, padding_mask=None):
        batch_size = ops.shape(inputs)[0]

        query = ops.expand_dims(self.pooling_attention_query, axis=0)
        query = ops.tile(query, [batch_size, 1, 1])

        if padding_mask is not None:
            padding_mask = ops.cast(padding_mask, "int32")
            padding_mask = ops.expand_dims(padding_mask, axis=[1, 2])

        x = self.pooling_attention(
            query=query, value=inputs, key=inputs, attention_mask=padding_mask
        )
        x = self.layer_norm(x)
        x = self.dropout_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "query_dim": self.query_dim,
                "num_queries": self.num_queries,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "use_per_dim_scale": self.use_per_dim_scale,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape, padding_mask_shape=None):
        query_shape = (inputs_shape[0], self.num_queries, self.query_dim)
        return query_shape
