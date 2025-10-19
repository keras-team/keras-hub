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


def _get_patches_center_coordinates(
    num_patches_h, num_patches_w, dtype="float32"
):
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


class DINOV3RopePositionEmbedding(layers.Layer):
    def __init__(self, hidden_dim, num_heads, rope_theta, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.patch_size = patch_size
        self.head_dim = hidden_dim // num_heads
        inv_freq = 1.0 / (
            rope_theta ** (ops.arange(0, 1, 4 / self.head_dim, dtype="float32"))
        )
        self.inv_freq = inv_freq

    def call(self, pixel_values):
        shape = ops.shape(pixel_values)
        height, width = shape[1], shape[2]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        patch_coords = _get_patches_center_coordinates(
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

        cos = ops.cos(angles)
        sin = ops.sin(angles)

        return ops.cast(cos, pixel_values.dtype), ops.cast(
            sin, pixel_values.dtype
        )

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


def _rotate_half(x):
    x1 = x[..., : ops.shape(x)[-1] // 2]
    x2 = x[..., ops.shape(x)[-1] // 2 :]
    return ops.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, num_prefix_tokens):
    q_prefix_tokens = q[:, :, :num_prefix_tokens, :]
    q_patches = q[:, :, num_prefix_tokens:, :]
    k_prefix_tokens = k[:, :, :num_prefix_tokens, :]
    k_patches = k[:, :, num_prefix_tokens:, :]

    cos = ops.expand_dims(ops.expand_dims(cos, axis=0), axis=0)
    sin = ops.expand_dims(ops.expand_dims(sin, axis=0), axis=0)

    q_patches = (q_patches * cos) + (_rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (_rotate_half(k_patches) * sin)

    q = ops.concatenate([q_prefix_tokens, q_patches], axis=-2)
    k = ops.concatenate([k_prefix_tokens, k_patches], axis=-2)

    return q, k


class DINOV3Attention(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        query_bias=True,
        key_bias=True,
        value_bias=True,
        proj_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = layers.Dense(
            hidden_dim, use_bias=query_bias, name="q_proj"
        )
        self.k_proj = layers.Dense(hidden_dim, use_bias=key_bias, name="k_proj")
        self.v_proj = layers.Dense(
            hidden_dim, use_bias=value_bias, name="v_proj"
        )
        self.o_proj = layers.Dense(
            hidden_dim, use_bias=proj_bias, name="o_proj"
        )
        self.dropout = layers.Dropout(dropout_rate)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        num_prefix_tokens=0,
    ):
        batch_size, seq_len, _ = ops.shape(hidden_states)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb(q, k, cos, sin, num_prefix_tokens)

        attn_weights = (
            ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        )

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = ops.matmul(attn_weights, v)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (batch_size, seq_len, -1))
        attn_output = self.o_proj(attn_output)

        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "query_bias": self.q_proj.use_bias,
                "key_bias": self.k_proj.use_bias,
                "value_bias": self.v_proj.use_bias,
                "proj_bias": self.o_proj.use_bias,
            }
        )
        return config


class DINOV3LayerScale(layers.Layer):
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
    def __init__(self, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)

    def build(self, input_shape):
        self.noise_shape = (input_shape[0],) + (1,) * (len(input_shape) - 1)

    def call(self, inputs, training=None):
        if not training or self.rate == 0.0:
            return inputs

        keep_prob = 1.0 - self.rate
        random_tensor = keep_prob + random.uniform(
            self.noise_shape, dtype=inputs.dtype
        )
        random_tensor = ops.floor(random_tensor)
        return ops.multiply(ops.divide(inputs, keep_prob), random_tensor)

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


class DINOV3MLP(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation="gelu",
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.use_bias = use_bias
        self.up_proj = layers.Dense(
            intermediate_dim, use_bias=use_bias, name="up_proj"
        )
        self.down_proj = layers.Dense(
            hidden_dim, use_bias=use_bias, name="down_proj"
        )
        self.act_fn = layers.Activation(activation)

    def call(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

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


class DINOV3GatedMLP(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation="gelu",
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        self.use_bias = use_bias
        self.gate_proj = layers.Dense(
            intermediate_dim, use_bias=use_bias, name="gate_proj"
        )
        self.up_proj = layers.Dense(
            intermediate_dim, use_bias=use_bias, name="up_proj"
        )
        self.down_proj = layers.Dense(
            hidden_dim, use_bias=use_bias, name="down_proj"
        )
        self.act_fn = layers.Activation(activation)

    def call(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

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


class DINOV3Layer(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        layer_scale_init_value=1.0,
        use_gated_mlp=False,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        query_bias=True,
        key_bias=True,
        value_bias=True,
        proj_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.layer_scale_init_value = layer_scale_init_value
        self.use_gated_mlp = use_gated_mlp
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps

        self.norm1 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="norm1"
        )
        self.attention = DINOV3Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=attention_dropout,
            query_bias=query_bias,
            key_bias=key_bias,
            value_bias=value_bias,
            proj_bias=proj_bias,
            name="attention",
        )
        self.layer_scale1 = DINOV3LayerScale(
            hidden_dim,
            init_values=layer_scale_init_value,
            name="layer_scale1",
        )
        self.drop_path = (
            DINOV3DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else layers.Identity()
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=layer_norm_eps, name="norm2"
        )
        if use_gated_mlp:
            self.mlp = DINOV3GatedMLP(hidden_dim, intermediate_dim, name="mlp")
        else:
            self.mlp = DINOV3MLP(hidden_dim, intermediate_dim, name="mlp")
        self.layer_scale2 = DINOV3LayerScale(
            hidden_dim, init_values=layer_scale_init_value, name="layer_scale2"
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        num_prefix_tokens=0,
    ):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            num_prefix_tokens=num_prefix_tokens,
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = self.drop_path(hidden_states) + residual

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_gated_mlp": self.use_gated_mlp,
                "attention_dropout": self.attention_dropout,
                "drop_path_rate": self.drop_path_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


class DINOV3Encoder(layers.Layer):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        intermediate_dim,
        layer_scale_init_value=1.0,
        use_gated_mlp=False,
        attention_dropout=0.0,
        drop_path_rate=0.0,
        layer_norm_eps=1e-6,
        query_bias=True,
        key_bias=True,
        value_bias=True,
        proj_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.layer_scale_init_value = layer_scale_init_value
        self.use_gated_mlp = use_gated_mlp
        self.attention_dropout = attention_dropout
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps

        dpr = [x for x in ops.linspace(0.0, drop_path_rate, num_layers)]
        self.layers = [
            DINOV3Layer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
                use_gated_mlp=use_gated_mlp,
                attention_dropout=attention_dropout,
                drop_path_rate=dpr[i],
                layer_norm_eps=layer_norm_eps,
                query_bias=query_bias,
                key_bias=key_bias,
                value_bias=value_bias,
                proj_bias=proj_bias,
                name=f"layers.{i}",
            )
            for i in range(num_layers)
        ]

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        num_prefix_tokens=0,
    ):
        pyramid_outputs = {}
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                num_prefix_tokens=num_prefix_tokens,
            )
            pyramid_outputs[f"stage{i + 1}"] = hidden_states

        return hidden_states, pyramid_outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_gated_mlp": self.use_gated_mlp,
                "attention_dropout": self.attention_dropout,
                "drop_path_rate": self.drop_path_rate,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config
