import math

from keras import initializers
from keras import layers
from keras import ops
from keras import random

from keras_hub.src.utils.keras_utils import standardize_data_format


class DINOV3PatchEmbedding(layers.Layer):
    """A layer that converts images into patches.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        patch_size: int. The size of one side of each patch.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, patch_size, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.data_format = standardize_data_format(data_format)

        self.projection = layers.Conv2D(
            hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            data_format=data_format,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="projection",
        )

    def build(self, input_shape):
        self.projection.build(input_shape)

    def call(self, inputs, training=None):
        batch_size = ops.shape(inputs)[0]
        embeddings = self.projection(inputs, training=training)
        if self.data_format == "channels_last":
            embeddings = ops.reshape(
                embeddings, (batch_size, -1, self.hidden_dim)
            )
        else:
            embeddings = ops.reshape(
                embeddings, (batch_size, self.hidden_dim, -1)
            )
            embeddings = ops.transpose(embeddings, (0, 2, 1))
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                output_shape[1] = patch_num**2
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                output_shape[1] = patch_num**2
        return output_shape


class DINOV3Embedding(layers.Layer):
    """A layer that converts images into patches.

    This layer adds all the necessary tokens to the embeddings, inlcuding
    the class token, register tokens and mask token if specified.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        patch_size: int. The size of one side of each patch.
        num_register_tokens: int. The number of register tokens to add to the
            embeddings. Defaults to `0`.
        use_mask_token: bool. Whether to use a mask token. Defaults to `True`.
        initializer_range: float. The standard deviation of the truncated
            normal initializer. Defaults to `0.02`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        patch_size,
        num_register_tokens=0,
        use_mask_token=True,
        initializer_range=0.02,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.num_register_tokens = int(num_register_tokens)
        self.use_mask_token = bool(use_mask_token)
        self.initializer_range = float(initializer_range)
        self.data_format = standardize_data_format(data_format)

        self.patch_embeddings = DINOV3PatchEmbedding(
            hidden_dim,
            patch_size,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="patch_embeddings",
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            trainable=True,
            name="cls_token",
        )
        if self.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, 1, self.hidden_dim),
                initializer="zeros",
                trainable=True,
                name="mask_token",
            )
        if self.num_register_tokens > 0:
            self.register_tokens = self.add_weight(
                shape=(1, self.num_register_tokens, self.hidden_dim),
                initializer=initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                trainable=True,
                name="register_tokens",
            )
        self.patch_embeddings.build(input_shape)

    def call(self, inputs, masks=None, training=None):
        batch_size = ops.shape(inputs)[0]
        embeddings = self.patch_embeddings(inputs, training=training)

        if masks is not None and self.use_mask_token:
            mask_token = ops.cast(self.mask_token, embeddings.dtype)
            embeddings = ops.where(
                ops.expand_dims(masks, axis=-1),
                mask_token,
                embeddings,
            )

        cls_tokens = ops.tile(self.cls_token, (batch_size, 1, 1))
        embeddings = ops.concatenate((cls_tokens, embeddings), axis=1)

        if self.num_register_tokens > 0:
            register_tokens = ops.tile(self.register_tokens, (batch_size, 1, 1))
            embeddings = ops.concatenate(
                (
                    embeddings[:, :1, ...],
                    register_tokens,
                    embeddings[:, 1:, ...],
                ),
                axis=1,
            )
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "num_register_tokens": self.num_register_tokens,
                "use_mask_token": self.use_mask_token,
                "initializer_range": self.initializer_range,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                output_shape[1] = 1 + self.num_register_tokens + patch_num**2
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                output_shape[1] = 1 + self.num_register_tokens + patch_num**2
        return output_shape


class DINOV3RopePositionEmbedding(layers.Layer):
    """A layer that implements Rotary Position Embedding.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        rope_theta: float. The base period of the rotary position embeddings.
        patch_size: int. The size of one side of each patch.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        rope_theta,
        patch_size,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.rope_theta = float(rope_theta)
        self.patch_size = int(patch_size)
        self.data_format = standardize_data_format(data_format)
        self.head_dim = hidden_dim // num_heads
        self.inv_freq = 1.0 / (
            rope_theta ** (ops.arange(0, 1, 4 / self.head_dim, dtype="float32"))
        )

    def _get_patches_center_coordinates(
        self, num_patches_h, num_patches_w, dtype="float32"
    ):
        """A helper function to get the center coordinates of the patches."""
        coords_h = ops.arange(0.5, num_patches_h, dtype=dtype)
        coords_w = ops.arange(0.5, num_patches_w, dtype=dtype)

        coords_h = coords_h / num_patches_h
        coords_w = coords_w / num_patches_w

        coords_h = ops.expand_dims(coords_h, axis=1)
        coords_w = ops.expand_dims(coords_w, axis=0)

        coords_h = ops.repeat(coords_h, num_patches_w, axis=1)
        coords_w = ops.repeat(coords_w, num_patches_h, axis=0)

        coords = ops.stack([coords_h, coords_w], axis=-1)
        coords = ops.reshape(coords, (-1, 2))
        coords = 2.0 * coords - 1.0
        return coords

    def call(self, inputs):
        shape = ops.shape(inputs)
        if self.data_format == "channels_last":
            height, width = shape[1], shape[2]
        else:
            height, width = shape[2], shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        patch_coords = self._get_patches_center_coordinates(
            num_patches_h, num_patches_w, dtype="float32"
        )
        angles = (
            2
            * math.pi
            * ops.expand_dims(patch_coords, axis=-1)
            * ops.expand_dims(ops.expand_dims(self.inv_freq, axis=0), axis=0)
        )
        angles = ops.reshape(angles, (ops.shape(angles)[0], -1))
        angles = ops.tile(angles, (1, 2))

        cos = ops.cast(ops.cos(angles), inputs.dtype)
        sin = ops.cast(ops.sin(angles), inputs.dtype)
        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "rope_theta": self.rope_theta,
                "patch_size": self.patch_size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.data_format == "channels_last":
            height, width = input_shape[1], input_shape[2]
        else:
            height, width = input_shape[2], input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        seq_len = num_patches_h * num_patches_w
        output_shape = (seq_len, self.head_dim)
        return output_shape, output_shape


class DINOV3Attention(layers.Layer):
    """A multi-head attention layer with dropout.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        dropout_rate: float. The dropout rate to use. Defaults to `0.0`.
        use_query_bias: bool. Whether to use a bias for the query projection.
        use_key_bias: bool. Whether to use a bias for the key projection.
        use_value_bias: bool. Whether to use a bias for the value projection.
        use_proj_bias: bool. Whether to use a bias for the output projection.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        use_query_bias=True,
        use_key_bias=True,
        use_value_bias=True,
        use_proj_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.use_query_bias = bool(use_query_bias)
        self.use_key_bias = bool(use_key_bias)
        self.use_value_bias = bool(use_value_bias)
        self.use_proj_bias = bool(use_proj_bias)
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query_dense = layers.Dense(
            hidden_dim,
            use_bias=use_query_bias,
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.key_dense = layers.Dense(
            hidden_dim,
            use_bias=use_key_bias,
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.value_dense = layers.Dense(
            hidden_dim,
            use_bias=use_value_bias,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.output_dense = layers.Dense(
            hidden_dim,
            use_bias=use_proj_bias,
            dtype=self.dtype_policy,
            name="o_proj",
        )
        self.dropout = layers.Dropout(
            dropout_rate, dtype=self.dtype_policy, name="dropout"
        )

    def build(self, input_shape):
        self.query_dense.build(input_shape)
        self.key_dense.build(input_shape)
        self.value_dense.build(input_shape)
        self.output_dense.build(input_shape)

    def _apply_rotary(self, q, k, cos, sin, num_prefix_tokens):
        """Apply rotary position embedding to query and key."""

        def _rotate_half(x):
            """A helper function to rotate half of the features."""
            x1 = x[..., : ops.shape(x)[-1] // 2]
            x2 = x[..., ops.shape(x)[-1] // 2 :]
            return ops.concatenate([-x2, x1], axis=-1)

        q_prefix_tokens = q[:, :num_prefix_tokens, :, :]
        q_patches = q[:, num_prefix_tokens:, :, :]
        k_prefix_tokens = k[:, :num_prefix_tokens, :, :]
        k_patches = k[:, num_prefix_tokens:, :, :]
        cos = ops.expand_dims(ops.expand_dims(cos, axis=0), axis=2)
        sin = ops.expand_dims(ops.expand_dims(sin, axis=0), axis=2)

        q_patches = (q_patches * cos) + (_rotate_half(q_patches) * sin)
        k_patches = (k_patches * cos) + (_rotate_half(k_patches) * sin)
        q = ops.concatenate([q_prefix_tokens, q_patches], axis=-3)
        k = ops.concatenate([k_prefix_tokens, k_patches], axis=-3)
        return q, k

    def call(
        self,
        inputs,
        attention_mask=None,
        position_embeddings=None,
        num_prefix_tokens=0,
        training=None,
    ):
        batch_size, seq_len, _ = ops.shape(inputs)
        q = self.query_dense(inputs, training=training)
        k = self.key_dense(inputs, training=training)
        v = self.value_dense(inputs, training=training)
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_rotary(q, k, cos, sin, num_prefix_tokens)

        attn_output = ops.nn.dot_product_attention(
            q,
            k,
            v,
            mask=attention_mask,
            scale=self.scale,
            is_causal=False,
        )
        attn_output = ops.reshape(attn_output, (batch_size, seq_len, -1))
        attn_output = self.dropout(attn_output, training=training)
        return self.output_dense(attn_output, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "query_bias": self.use_query_bias,
                "key_bias": self.use_key_bias,
                "value_bias": self.use_value_bias,
                "proj_bias": self.use_proj_bias,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DINOV3LayerScale(layers.Layer):
    """A layer scale.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        init_values: float. The initial value for the scale. Defaults to `1.0`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, init_values=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.init_values = float(init_values)

    def build(self, input_shape):
        self.lambda1 = self.add_weight(
            shape=(self.hidden_dim,),
            initializer=initializers.Constant(self.init_values),
            trainable=True,
            name="lambda1",
        )

    def call(self, inputs, training=None):
        return ops.multiply(inputs, self.lambda1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_dim": self.hidden_dim, "init_values": self.init_values}
        )
        return config


class DINOV3DropPath(layers.Layer):
    """A drop path layer.

    Args:
        rate: float. The drop path rate to use. Defaults to `0.0`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def build(self, input_shape):
        self.noise_shape = (input_shape[0],) + (1,) * (len(input_shape) - 1)

    def call(self, inputs, training=None):
        if not training or self.rate == 0.0:
            return inputs

        keep_prob = 1.0 - self.rate
        random_tensor = random.uniform(self.noise_shape, dtype=inputs.dtype)
        random_tensor = ops.add(random_tensor, keep_prob)
        return ops.multiply(ops.divide(inputs, keep_prob), random_tensor)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DINOV3MLP(layers.Layer):
    """A DINOV3 MLP block.

    Args:
        hidden_dim: int. The number of units in the output layer.
        intermediate_dim: int. The output dimension of the first Dense layer.
        activation: str of callable. Activation to use in the intermediate
            layer. Defaults to `"gelu"`.
        use_bias: bool. Whether to use a bias for the dense layers.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation="gelu",
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.activation = activation
        self.use_bias = bool(use_bias)

        self.up_proj = layers.Dense(
            intermediate_dim,
            activation=activation,
            use_bias=use_bias,
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.down_proj = layers.Dense(
            hidden_dim,
            use_bias=use_bias,
            dtype=self.dtype_policy,
            name="down_proj",
        )

    def build(self, input_shape):
        self.up_proj.build(input_shape)
        input_shape = self.up_proj.compute_output_shape(input_shape)
        self.down_proj.build(input_shape)

    def call(self, inputs, training=None):
        x = self.up_proj(inputs, training=training)
        return self.down_proj(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape


class DINOV3GatedMLP(layers.Layer):
    """A DINOV3 Gated MLP block.

    Args:
        hidden_dim: int. The number of units in the output layer.
        intermediate_dim: int. The output dimension of the first Dense layer.
        activation: str of callable. Activation to use in the intermediate
            layer. Defaults to `"gelu"`.
        use_bias: bool. Whether to use a bias for the dense layers.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation="gelu",
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.activation = activation
        self.use_bias = bool(use_bias)

        self.gate_proj = layers.Dense(
            intermediate_dim,
            activation=activation,
            use_bias=use_bias,
            dtype=self.dtype_policy,
            name="gate_proj",
        )
        self.up_proj = layers.Dense(
            intermediate_dim,
            use_bias=use_bias,
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.down_proj = layers.Dense(
            hidden_dim,
            use_bias=use_bias,
            dtype=self.dtype_policy,
            name="down_proj",
        )

    def build(self, input_shape):
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        input_shape = self.up_proj.compute_output_shape(input_shape)
        self.down_proj.build(input_shape)

    def call(self, inputs, training=None):
        x = ops.multiply(
            self.gate_proj(inputs, training=training),
            self.up_proj(inputs, training=training),
        )
        return self.down_proj(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape


class DINOV3Layer(layers.Layer):
    """A DINOV3 encoder layer.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        layer_scale_init_value: float. The initial value for the scale.
            Defaults to `1.0`.
        hidden_activation: str or callable. Activation to use in the MLP.
            Defaults to `"gelu"`.
        use_gated_mlp: bool. Whether to use Gated MLP layers. Defaults to
            `False`.
        use_query_bias: bool. Whether to use a bias for the query projection.
        use_key_bias: bool. Whether to use a bias for the key projection.
        use_value_bias: bool. Whether to use a bias for the value projection.
        use_proj_bias: bool. Whether to use a bias for the output projection.
        use_mlp_bias: bool. Whether to use a bias for the MLP layers.
        attention_dropout: float. The dropout rate for the attention
            probabilities. Defaults to `0.0`.
        drop_path_rate: float. The drop path rate to use. Defaults to `0.0`.
        layer_norm_eps: float. The epsilon for layer normalization.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        layer_scale_init_value=1.0,
        hidden_activation="gelu",
        use_gated_mlp=False,
        use_query_bias=True,
        use_key_bias=True,
        use_value_bias=True,
        use_proj_bias=True,
        use_mlp_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.hidden_activation = hidden_activation
        self.use_gated_mlp = bool(use_gated_mlp)
        self.use_query_bias = bool(use_query_bias)
        self.use_key_bias = bool(use_key_bias)
        self.use_value_bias = bool(use_value_bias)
        self.use_proj_bias = bool(use_proj_bias)
        self.use_mlp_bias = bool(use_mlp_bias)
        self.attention_dropout = float(attention_dropout)
        self.drop_path_rate = float(drop_path_rate)
        self.layer_norm_eps = float(layer_norm_eps)

        self.norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps, dtype=self.dtype_policy, name="norm1"
        )
        self.attention = DINOV3Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=attention_dropout,
            use_query_bias=use_query_bias,
            use_key_bias=use_key_bias,
            use_value_bias=use_value_bias,
            use_proj_bias=use_proj_bias,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_scale1 = DINOV3LayerScale(
            hidden_dim,
            init_values=layer_scale_init_value,
            dtype=self.dtype_policy,
            name="layer_scale1",
        )
        self.drop_path = (
            DINOV3DropPath(drop_path_rate, dtype=self.dtype_policy)
            if drop_path_rate > 0.0
            else layers.Identity(dtype=self.dtype_policy)
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps, dtype=self.dtype_policy, name="norm2"
        )
        if use_gated_mlp:
            self.mlp = DINOV3GatedMLP(
                hidden_dim,
                intermediate_dim,
                activation=hidden_activation,
                use_bias=use_mlp_bias,
                dtype=self.dtype_policy,
                name="mlp",
            )
        else:
            self.mlp = DINOV3MLP(
                hidden_dim,
                intermediate_dim,
                activation=hidden_activation,
                use_bias=use_mlp_bias,
                dtype=self.dtype_policy,
                name="mlp",
            )
        self.layer_scale2 = DINOV3LayerScale(
            hidden_dim,
            init_values=layer_scale_init_value,
            dtype=self.dtype_policy,
            name="layer_scale2",
        )

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.attention.build(input_shape)
        input_shape = self.attention.compute_output_shape(input_shape)
        self.layer_scale1.build(input_shape)
        self.drop_path.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        input_shape = self.mlp.compute_output_shape(input_shape)
        self.layer_scale2.build(input_shape)

    def call(
        self,
        inputs,
        attention_mask=None,
        position_embeddings=None,
        num_prefix_tokens=0,
        training=None,
    ):
        residual = inputs
        hidden_states = self.norm1(inputs)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            num_prefix_tokens=num_prefix_tokens,
            training=training,
        )
        hidden_states = self.layer_scale1(hidden_states, training=training)
        hidden_states = (
            self.drop_path(hidden_states, training=training) + residual
        )

        residual = hidden_states
        hidden_states = self.norm2(hidden_states, training=training)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = self.layer_scale2(hidden_states, training=training)
        return self.drop_path(hidden_states, training=training) + residual

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_scale_init_value": self.layer_scale_init_value,
                "hidden_activation": self.hidden_activation,
                "use_gated_mlp": self.use_gated_mlp,
                "use_query_bias": self.use_query_bias,
                "use_key_bias": self.use_key_bias,
                "use_value_bias": self.use_value_bias,
                "use_proj_bias": self.use_proj_bias,
                "use_mlp_bias": self.use_mlp_bias,
                "attention_dropout": self.attention_dropout,
                "drop_path_rate": self.drop_path_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DINOV3Encoder(layers.Layer):
    """A DINOV3 encoder.

    Args:
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        layer_scale_init_value: float. The initial value for the scale.
            Defaults to `1.0`.
        hidden_activation: str or callable. Activation to use in the MLP.
            Defaults to `"gelu"`.
        use_gated_mlp: bool. Whether to use Gated MLP layers. Defaults to
            `False`.
        use_query_bias: bool. Whether to use a bias for the query projection.
            Defaults to `True`.
        use_key_bias: bool. Whether to use a bias for the key projection.
            Defaults to `True`.
        use_value_bias: bool. Whether to use a bias for the value projection.
            Defaults to `True`.
        use_proj_bias: bool. Whether to use a bias for the output projection.
            Defaults to `True`.
        use_mlp_bias: bool. Whether to use a bias for the dense layers in MLP.
            Defaults to `True`.
        attention_dropout: float. The dropout rate for the attention
            probabilities. Defaults to `0.0`.
        drop_path_rate: float. The drop path rate to use. Defaults to `0.0`.
        layer_norm_eps: float. The epsilon for layer normalization. Defaults to
            `1e-5`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        intermediate_dim,
        layer_scale_init_value=1.0,
        hidden_activation="gelu",
        use_gated_mlp=False,
        use_query_bias=True,
        use_key_bias=True,
        use_value_bias=True,
        use_proj_bias=True,
        use_mlp_bias=True,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.hidden_activation = hidden_activation
        self.use_gated_mlp = bool(use_gated_mlp)
        self.use_query_bias = bool(use_query_bias)
        self.use_key_bias = bool(use_key_bias)
        self.use_value_bias = bool(use_value_bias)
        self.use_proj_bias = bool(use_proj_bias)
        self.use_mlp_bias = bool(use_mlp_bias)
        self.attention_dropout = float(attention_dropout)
        self.drop_path_rate = float(drop_path_rate)
        self.layer_norm_eps = float(layer_norm_eps)

        dpr = [x for x in ops.linspace(0.0, drop_path_rate, num_layers)]
        self.layers = [
            DINOV3Layer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                hidden_activation=hidden_activation,
                use_gated_mlp=use_gated_mlp,
                use_query_bias=use_query_bias,
                use_key_bias=use_key_bias,
                use_value_bias=use_value_bias,
                use_proj_bias=use_proj_bias,
                use_mlp_bias=use_mlp_bias,
                attention_dropout=attention_dropout,
                drop_path_rate=dpr[i],
                layer_norm_eps=layer_norm_eps,
                dtype=self.dtype_policy,
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def call(
        self,
        inputs,
        attention_mask=None,
        position_embeddings=None,
        num_prefix_tokens=0,
        training=None,
    ):
        pyramid_outputs = {}
        x = inputs
        for layer_index, layer in enumerate(self.layers, start=1):
            x = layer(
                x,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                num_prefix_tokens=num_prefix_tokens,
                training=training,
            )
            pyramid_outputs[f"stage{str(layer_index)}"] = x
        return x, pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_scale_init_value": self.layer_scale_init_value,
                "hidden_activation": self.hidden_activation,
                "use_gated_mlp": self.use_gated_mlp,
                "use_query_bias": self.use_query_bias,
                "use_key_bias": self.use_key_bias,
                "use_value_bias": self.use_value_bias,
                "use_proj_bias": self.use_proj_bias,
                "use_mlp_bias": self.use_mlp_bias,
                "attention_dropout": self.attention_dropout,
                "drop_path_rate": self.drop_path_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        pyramid_outputs = {}
        for layer_index in range(1, len(self.layers) + 1):
            pyramid_outputs[f"stage{str(layer_index)}"] = input_shape
        return input_shape, pyramid_outputs
