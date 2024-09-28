import math

from keras import layers
from keras import ops

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import standardize_data_format


class VAEAttention(layers.Layer):
    def __init__(self, filters, groups=32, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.data_format = standardize_data_format(data_format)
        gn_axis = -1 if self.data_format == "channels_last" else 1

        self.group_norm = layers.GroupNormalization(
            groups=groups,
            axis=gn_axis,
            epsilon=1e-6,
            dtype="float32",
            name="group_norm",
        )
        self.query_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="query_conv2d",
        )
        self.key_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="key_conv2d",
        )
        self.value_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="value_conv2d",
        )
        self.softmax = layers.Softmax(dtype="float32")
        self.output_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="output_conv2d",
        )

        self.groups = groups
        self._inverse_sqrt_filters = 1.0 / math.sqrt(float(filters))

    def build(self, input_shape):
        self.group_norm.build(input_shape)
        self.query_conv2d.build(input_shape)
        self.key_conv2d.build(input_shape)
        self.value_conv2d.build(input_shape)
        self.output_conv2d.build(input_shape)

    def call(self, inputs, training=None):
        x = self.group_norm(inputs)
        query = self.query_conv2d(x)
        key = self.key_conv2d(x)
        value = self.value_conv2d(x)

        if self.data_format == "channels_first":
            query = ops.transpose(query, (0, 2, 3, 1))
            key = ops.transpose(key, (0, 2, 3, 1))
            value = ops.transpose(value, (0, 2, 3, 1))
        shape = ops.shape(inputs)
        b = shape[0]
        query = ops.reshape(query, (b, -1, self.filters))
        key = ops.reshape(key, (b, -1, self.filters))
        value = ops.reshape(value, (b, -1, self.filters))

        # Compute attention.
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_filters, query.dtype)
        )
        # [B, H0 * W0, C], [B, H1 * W1, C] -> [B, H0 * W0, H1 * W1]
        attention_scores = ops.einsum("abc,adc->abd", query, key)
        attention_scores = ops.cast(
            self.softmax(attention_scores), self.compute_dtype
        )
        # [B, H2 * W2, C], [B, H0 * W0, H1 * W1] -> [B, H1 * W1 ,C]
        attention_output = ops.einsum("abc,adb->adc", value, attention_scores)
        x = ops.reshape(attention_output, shape)

        x = self.output_conv2d(x)
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
        x = ops.add(x, inputs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "groups": self.groups,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def apply_resnet_block(x, filters, data_format=None, dtype=None, name=None):
    data_format = standardize_data_format(data_format)
    gn_axis = -1 if data_format == "channels_last" else 1
    input_filters = x.shape[gn_axis]

    residual = x
    x = layers.GroupNormalization(
        groups=32,
        axis=gn_axis,
        epsilon=1e-6,
        dtype="float32",
        name=f"{name}_norm1",
    )(x)
    x = layers.Activation("swish", dtype=dtype)(x)
    x = layers.Conv2D(
        filters,
        3,
        1,
        padding="same",
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_conv1",
    )(x)
    x = layers.GroupNormalization(
        groups=32,
        axis=gn_axis,
        epsilon=1e-6,
        dtype="float32",
        name=f"{name}_norm2",
    )(x)
    x = layers.Activation("swish", dtype=dtype)(x)
    x = layers.Conv2D(
        filters,
        3,
        1,
        padding="same",
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_conv2",
    )(x)
    if input_filters != filters:
        residual = layers.Conv2D(
            filters,
            1,
            1,
            data_format=data_format,
            dtype=dtype,
            name=f"{name}_residual_projection",
        )(residual)
    x = layers.Add(dtype=dtype)([residual, x])
    return x


class VAEImageDecoder(Backbone):
    """Decoder for the VAE model used in Stable Diffusion 3.

    Args:
        stackwise_num_filters: list of ints. The number of filters for each
            stack.
        stackwise_num_blocks: list of ints. The number of blocks for each stack.
        output_channels: int. The number of channels in the output.
        latent_shape: tuple. The shape of the latent image.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.
    """

    def __init__(
        self,
        stackwise_num_filters,
        stackwise_num_blocks,
        output_channels=3,
        latent_shape=(None, None, 16),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        gn_axis = -1 if data_format == "channels_last" else 1

        # === Functional Model ===
        latent_inputs = layers.Input(shape=latent_shape)

        x = layers.Conv2D(
            stackwise_num_filters[0],
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name="input_projection",
        )(latent_inputs)
        x = apply_resnet_block(
            x,
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=dtype,
            name="input_block0",
        )
        x = VAEAttention(
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=dtype,
            name="input_attention",
        )(x)
        x = apply_resnet_block(
            x,
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=dtype,
            name="input_block1",
        )

        # Stacks.
        for i, filters in enumerate(stackwise_num_filters):
            for j in range(stackwise_num_blocks[i]):
                x = apply_resnet_block(
                    x,
                    filters,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"block{i}_{j}",
                )
            if i != len(stackwise_num_filters) - 1:
                # No upsamling in the last blcok.
                x = layers.UpSampling2D(
                    2,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"upsample_{i}",
                )(x)
                x = layers.Conv2D(
                    filters,
                    3,
                    1,
                    padding="same",
                    data_format=data_format,
                    dtype=dtype,
                    name=f"upsample_{i}_conv",
                )(x)

        # Ouput block.
        x = layers.GroupNormalization(
            groups=32,
            axis=gn_axis,
            epsilon=1e-6,
            dtype="float32",
            name="output_norm",
        )(x)
        x = layers.Activation("swish", dtype=dtype, name="output_activation")(x)
        image_outputs = layers.Conv2D(
            output_channels,
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=dtype,
            name="output_projection",
        )(x)
        super().__init__(inputs=latent_inputs, outputs=image_outputs, **kwargs)

        # === Config ===
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_num_blocks = stackwise_num_blocks
        self.output_channels = output_channels
        self.latent_shape = latent_shape

    @property
    def scaling_factor(self):
        """The scaling factor for the latent space.

        This is used to scale the latent space to have unit variance when
        training the diffusion model.
        """
        return 1.5305

    @property
    def shift_factor(self):
        """The shift factor for the latent space.

        This is used to shift the latent space to have zero mean when
        training the diffusion model.
        """
        return 0.0609

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_num_blocks": self.stackwise_num_blocks,
                "output_channels": self.output_channels,
                "image_shape": self.latent_shape,
            }
        )
        return config
