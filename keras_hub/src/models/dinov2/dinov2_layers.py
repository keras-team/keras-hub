import keras
from keras import backend
from keras import config
from keras import initializers
from keras import layers
from keras import ops
from keras import random

from keras_hub.src.utils.keras_utils import standardize_data_format


class DINOV2PatchEmbedding(layers.Layer):
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


class DINOV2Embedding(layers.Layer):
    """A layer that converts images into patches.

    This layer adds all the necessary tokens to the embeddings, inlcuding
    the class token, register tokens and mask token if specified. Finally, a
    position embedding will be added.

    This layer supports the interpolation of the position embeddings to enable
    the model to work with images of different sizes. Please refer to
    `_interpolate_position_embeddings` for more details.

    The saving and loading of this layer will automatically handle the position
    embeddings interpolation. Please refer to `save_own_variables` and
    `load_own_variables` for more details.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        patch_size: int. The size of one side of each patch.
        image_size: tuple of ints. The (height, width) of the input images.
        num_register_tokens: int. The number of register tokens to add to the
            embeddings. Defaults to `0`.
        use_mask_token: bool. Whether to use a mask token. Defaults to `True`.
        dropout_rate: float. The dropout rate to use. Defaults to `0.0`.
        position_embedding_shape: tuple. The original input shape used to
            train the position embeddings. This is used to interpolate the
            position embeddings to the actual input shape. Defaults to
            `(518, 518)`.
        antialias_in_interpolation: bool. Whether to use antialiasing in the
            interpolation of the position embeddings. Defaults to `False`.
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
        image_shape,
        num_register_tokens=0,
        use_mask_token=True,
        dropout_rate=0.0,
        position_embedding_shape=(518, 518),
        antialias_in_interpolation=False,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.position_embedding_shape = (
            int(position_embedding_shape[0]),
            int(position_embedding_shape[1]),
        )
        self.num_register_tokens = int(num_register_tokens)
        self.use_mask_token = bool(use_mask_token)
        self.dropout_rate = float(dropout_rate)
        self.antialias_in_interpolation = bool(antialias_in_interpolation)
        self.data_format = standardize_data_format(data_format)
        self.interpolated_num_patches = (
            self.image_shape[0] // self.patch_size
        ) * (self.image_shape[1] // self.patch_size)
        self.num_patches = (
            self.position_embedding_shape[0] // self.patch_size
        ) * (self.position_embedding_shape[1] // self.patch_size)

        self.patch_embeddings = DINOV2PatchEmbedding(
            hidden_dim,
            patch_size,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="patch_embeddings",
        )
        self.dropout = layers.Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="dropout",
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )
        if self.use_mask_token:
            self.mask_token = self.add_weight(
                shape=(1, self.hidden_dim),
                initializer="zeros",
                trainable=True,
                name="mask_token",
            )
        if self.num_register_tokens > 0:
            self.register_tokens = self.add_weight(
                shape=(1, self.num_register_tokens, self.hidden_dim),
                initializer="zeros",
                trainable=True,
                name="register_tokens",
            )
        self.patch_embeddings.build(input_shape)

        # Note that there are two position embeddings:
        # `self.interpolated_position_embeddings` is used for the image inputs
        # during both training and inference.
        # `self.position_embeddings` is used to load pretrained weights and
        # remains unchanged during training and inference. It will be updated
        # during saving once `self.interpolated_position_embeddings` is
        # modified.
        self.position_embeddings = self.add_weight(
            shape=(1, self.num_patches + 1, self.hidden_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="position_embeddings",
        )
        self.interpolated_position_embeddings = self.add_weight(
            shape=(1, self.interpolated_num_patches + 1, self.hidden_dim),
            initializer="zeros",  # Will be initialized by interpolation.
            trainable=True,
            name="interpolated_position_embeddings",
        )

        # Initialize the interpolated position embeddings.
        self.interpolated_position_embeddings.assign(
            self._interpolate_position_embeddings(
                self.position_embeddings,
                patch_size=self.patch_size,
                source_shape=self.position_embedding_shape,
                target_shape=self.image_shape,
                antialias=self.antialias_in_interpolation,
            )
        )

    def call(self, inputs, masks=None, training=None):
        batch_size = ops.shape(inputs)[0]
        embeddings = self.patch_embeddings(inputs, training=training)

        # Repalce the embeddings with the mask tokens if specified.
        # Basically, this is only used during training.
        if masks is not None and self.use_mask_token:
            masks = ops.expand_dims(masks, axis=-1)
            mask_token = ops.cast(
                ops.expand_dims(self.mask_token, axis=0), embeddings.dtype
            )
            embeddings = ops.where(masks, mask_token, embeddings)

        # Add the [CLS] token to the embedded patch tokens.
        cls_tokens = ops.tile(self.cls_token, (batch_size, 1, 1))
        embeddings = ops.concatenate((cls_tokens, embeddings), axis=1)

        # Add positional encoding to each token.
        embeddings = ops.add(embeddings, self.interpolated_position_embeddings)

        # Add register tokens if specified.
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

        embeddings = self.dropout(embeddings)
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_shape": self.image_shape,
                "num_register_tokens": self.num_register_tokens,
                "use_mask_token": self.use_mask_token,
                "dropout_rate": self.dropout_rate,
                "position_embedding_shape": self.position_embedding_shape,
                "antialias_in_interpolation": self.antialias_in_interpolation,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], None, self.hidden_dim]
        if self.data_format == "channels_last":
            if input_shape[1] is not None and input_shape[2] is not None:
                patch_num = input_shape[1] // self.patch_size
                # 1 is for cls token.
                output_shape[1] = 1 + self.num_register_tokens + patch_num**2
        else:
            if input_shape[2] is not None and input_shape[3] is not None:
                patch_num = input_shape[2] // self.patch_size
                # 1 is for cls token.
                output_shape[1] = 1 + self.num_register_tokens + patch_num**2
        return output_shape

    def compute_output_spec(self, inputs):
        output_shape = self.compute_output_shape(inputs.shape)
        return keras.KerasTensor(output_shape, dtype=self.compute_dtype)

    @staticmethod
    def _interpolate_position_embeddings(
        position_embeddings,
        patch_size,
        source_shape,
        target_shape,
        antialias=False,
    ):
        """Interpolate position embeddings to match the target image shape.

        Reference:
            - https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py
        """
        position_embeddings = ops.convert_to_tensor(position_embeddings)
        patch_size = int(patch_size)
        source_shape = (int(source_shape[0]), int(source_shape[1]))
        target_shape = (int(target_shape[0]), int(target_shape[1]))
        hidden_dim = int(position_embeddings.shape[-1])

        if (
            source_shape[0] == target_shape[0]
            and source_shape[1] == target_shape[1]
        ):
            # No need to interpolate if the image size is the same as the
            # position embedding image size.
            return ops.copy(position_embeddings)

        num_positions = int(position_embeddings.shape[1]) - 1

        # Handle class token and patch embeddings separately.
        class_position_embeddings = position_embeddings[:, :1, ...]
        patch_position_embeddings = position_embeddings[:, 1:, ...]

        # Calculate new dimensions
        new_height = target_shape[0] // patch_size
        new_width = target_shape[1] // patch_size

        # Reshape for interpolation
        sqrt_num_positions = int(num_positions**0.5)
        patch_position_embeddings = ops.reshape(
            patch_position_embeddings,
            (1, sqrt_num_positions, sqrt_num_positions, hidden_dim),
        )

        # Interpolate at float32 precision.
        original_dtype = backend.standardize_dtype(
            patch_position_embeddings.dtype
        )
        interpolated_patch_position_embeddings = ops.image.resize(
            ops.cast(patch_position_embeddings, "float32"),
            size=(new_height, new_width),
            interpolation="bicubic",
            antialias=antialias,
            data_format="channels_last",
        )
        interpolated_patch_position_embeddings = ops.cast(
            interpolated_patch_position_embeddings, original_dtype
        )

        # Reshape back to the original format
        interpolated_patch_position_embeddings = ops.reshape(
            interpolated_patch_position_embeddings, (1, -1, hidden_dim)
        )
        interpolated_position_embeddings = ops.concatenate(
            (class_position_embeddings, interpolated_patch_position_embeddings),
            axis=1,
        )
        return interpolated_position_embeddings

    def _is_interpolated_position_embeddings_updated(self):
        """Check if the interpolated position embeddings are updated."""
        original_interpolated_position_embeddings = (
            self._interpolate_position_embeddings(
                self.position_embeddings,
                patch_size=self.patch_size,
                source_shape=self.position_embedding_shape,
                target_shape=self.image_shape,
                antialias=self.antialias_in_interpolation,
            )
        )
        diff = ops.sum(
            ops.subtract(
                original_interpolated_position_embeddings,
                self.interpolated_position_embeddings,
            )
        )
        return ops.cond(
            ops.greater(diff, config.epsilon()), lambda: True, lambda: False
        )

    def save_own_variables(self, store):
        if self._is_interpolated_position_embeddings_updated():
            self.position_embeddings.assign(
                self._interpolate_position_embeddings(
                    self.interpolated_position_embeddings,
                    patch_size=self.patch_size,
                    source_shape=self.image_shape,
                    target_shape=self.position_embedding_shape,
                    antialias=self.antialias_in_interpolation,
                )
            )
        super().save_own_variables(store)

    def load_own_variables(self, store):
        all_vars = self._trainable_variables + self._non_trainable_variables
        for i, v in enumerate(all_vars):
            if v is self.interpolated_position_embeddings:
                continue
            v.assign(store[f"{i}"])
        self.interpolated_position_embeddings.assign(
            self._interpolate_position_embeddings(
                self.position_embeddings,
                patch_size=self.patch_size,
                source_shape=self.position_embedding_shape,
                target_shape=self.image_shape,
                antialias=self.antialias_in_interpolation,
            )
        )


class DINOV2Attention(layers.Layer):
    """A multi-head attention layer with dropout.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        dropout_rate: float. The dropout rate to use. Defaults to `0.0`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, num_heads, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)

        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_dim // self.num_heads,
            dropout=self.dropout_rate,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.dropout = layers.Dropout(
            rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="dropout",
        )

    def build(self, input_shape):
        self.attention.build(input_shape, input_shape)

    def call(self, inputs, training=None):
        attention_output = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
            use_causal_mask=False,
        )
        outputs = self.dropout(attention_output, training=training)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DINOV2LayerScale(layers.Layer):
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

    def compute_output_shape(self, input_shape):
        return input_shape


class DINOV2DropPath(layers.Layer):
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


class DINOV2MLP(layers.Layer):
    """A DINOV2 MLP block.

    Args:
        hidden_dim: int. The number of units in the output layer.
        intermediate_dim: int. The output dimension of the first Dense layer.
        activation: str of callable. Activation to use in the intermediate
            layer. Defaults to `"gelu"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self, hidden_dim, intermediate_dim, activation="gelu", **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.activation = activation

        self.fc1 = layers.Dense(
            self.intermediate_dim,
            activation=activation,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="fc1",
        )
        self.fc2 = layers.Dense(
            self.hidden_dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="fc2",
        )

    def build(self, input_shape):
        self.fc1.build(input_shape)
        input_shape = self.fc1.compute_output_shape(input_shape)
        self.fc2.build(input_shape)

    def call(self, inputs, training=None):
        x = self.fc1(inputs, training=training)
        x = self.fc2(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "activation": self.activation,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape


class DINOV2SwiGLUFFN(layers.Layer):
    """A DINOV2 SwiGLU Feed-Forward Network layer.

    Please refer to [GLU Variants Improve Transformer](
    https://arxiv.org/abs/2002.05202) for more details on SwiGLU.

    Args:
        hidden_dim: int. The number of units in the output layer.
        intermediate_dim: int. The output dimension of the first Dense layer.
            Note that this value will be multiplied by `2 / 3` and rounded up to
            the nearest multiple of `8`. The reason for this is that SwiGLUFFN
            achieves similar or better performance with fewer parameters
            compared to the original FFN implementation.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, intermediate_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.actual_intermediate_dim = (
            (int(intermediate_dim * 2 / 3) + 7) // 8 * 8
        )

        self.weights_in = layers.Dense(
            2 * self.actual_intermediate_dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="weights_in",
        )
        self.weights_out = layers.Dense(
            self.hidden_dim,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            dtype=self.dtype_policy,
            name="weights_out",
        )

    def build(self, input_shape):
        self.weights_in.build(input_shape)
        input_shape = list(input_shape)
        input_shape[-1] = self.actual_intermediate_dim
        self.weights_out.build(input_shape)

    def call(self, inputs, training=None):
        x = self.weights_in(inputs, training=training)
        x1, x2 = ops.split(x, 2, axis=-1)
        x = ops.multiply(ops.silu(x1), x2)
        x = self.weights_out(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_dim
        return output_shape


class DINOV2Layer(layers.Layer):
    """A DINOV2 encoder layer.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        layer_scale_init_value: float. The initial value for the scale.
            Defaults to `1.0`.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        use_swiglu_ffn: bool. Whether to use SwigLUFFN instead of MLP.
            Defaults to `False`.
        dropout_rate: float. The dropout rate to use. Defaults to `0.0`.
        drop_path_rate: float. The drop path rate to use. Defaults to `0.0`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim,
        layer_scale_init_value=1.0,
        use_swiglu_ffn=False,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.use_swiglu_ffn = bool(use_swiglu_ffn)
        self.dropout_rate = float(dropout_rate)
        self.drop_path_rate = float(drop_path_rate)

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy, name="norm1"
        )
        self.attention = DINOV2Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.layer_scale1 = DINOV2LayerScale(
            hidden_dim=self.hidden_dim,
            init_values=self.layer_scale_init_value,
            dtype=self.dtype_policy,
            name="layer_scale1",
        )
        self.drop_path = (
            DINOV2DropPath(
                rate=self.drop_path_rate,
                dtype=self.dtype_policy,
                name="drop_path",
            )
            if self.drop_path_rate > 0
            else layers.Identity(dtype=self.dtype_policy, name="drop_path")
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy, name="norm2"
        )
        if self.use_swiglu_ffn:
            self.mlp = DINOV2SwiGLUFFN(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                dtype=self.dtype_policy,
                name="mlp",
            )
        else:
            self.mlp = DINOV2MLP(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                activation="gelu",
                dtype=self.dtype_policy,
                name="mlp",
            )
        self.layer_scale2 = DINOV2LayerScale(
            hidden_dim=self.hidden_dim,
            init_values=self.layer_scale_init_value,
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

    def call(self, inputs, training=None):
        x = inputs
        x = self.norm1(x, training=training)
        x = self.attention(x, training=training)
        x = self.layer_scale1(x, training=training)

        # First residual connection.
        hidden_states = ops.add(self.drop_path(x, training=training), inputs)
        x = self.norm2(hidden_states, training=training)
        x = self.mlp(x, training=training)
        x = self.layer_scale2(x, training=training)

        # Second residual connection.
        return ops.add(self.drop_path(x, training=training), hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_scale_init_value": self.layer_scale_init_value,
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "dropout_rate": self.dropout_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DINOV2Encoder(layers.Layer):
    """A DINOV2 encoder.

    Args:
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The number of units in the hidden layers.
        num_heads: int. Number of attention heads.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        layer_scale_init_value: float. The initial value for the scale.
            Defaults to `1.0`.
        use_swiglu_ffn: bool. Whether to use SwigLUFFN instead of MLP.
            Defaults to `False`.
        dropout_rate: float. The dropout rate to use. Defaults to `0.0`.
        drop_path_rate: float. The drop path rate to use. Defaults to `0.0`.
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
        use_swiglu_ffn=False,
        dropout_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(intermediate_dim)
        self.layer_scale_init_value = float(layer_scale_init_value)
        self.use_swiglu_ffn = bool(use_swiglu_ffn)
        self.dropout_rate = float(dropout_rate)
        self.drop_path_rate = float(drop_path_rate)

        self.layers = [
            DINOV2Layer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                intermediate_dim=self.intermediate_dim,
                layer_scale_init_value=self.layer_scale_init_value,
                use_swiglu_ffn=self.use_swiglu_ffn,
                dropout_rate=self.dropout_rate,
                drop_path_rate=self.drop_path_rate,
                dtype=self.dtype_policy,
                name=f"layers_{i}",
            )
            for i in range(self.num_layers)
        ]

    def build(self, input_shape):
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def call(self, inputs, training=None):
        pyramid_outputs = {}
        x = inputs
        for layer_index, layer in enumerate(self.layers, start=1):
            x = layer(x, training=training)
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
                "use_swiglu_ffn": self.use_swiglu_ffn,
                "dropout_rate": self.dropout_rate,
                "drop_path_rate": self.drop_path_rate,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        pyramid_outputs = {}
        for layer_index in range(1, len(self.layers) + 1):
            pyramid_outputs[f"stage{str(layer_index)}"] = input_shape
        return input_shape, pyramid_outputs
