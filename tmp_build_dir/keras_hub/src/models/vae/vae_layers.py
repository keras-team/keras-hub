import math

import keras
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


class Conv2DMultiHeadAttention(keras.layers.Layer):
    """A MultiHeadAttention layer utilizing `Conv2D` and `GroupNormalization`.

    Args:
        filters: int. The number of the filters for the convolutional layers.
        groups: int. The number of the groups for the group normalization
            layers. Defaults to `32`.
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

    def __init__(self, filters, groups=32, data_format=None, **kwargs):
        super().__init__(**kwargs)
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        self.filters = int(filters)
        self.groups = int(groups)
        self._inverse_sqrt_filters = 1.0 / math.sqrt(float(filters))
        self.data_format = data_format

        self.group_norm = keras.layers.GroupNormalization(
            groups=groups,
            axis=channel_axis,
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="group_norm",
        )
        self.query_conv2d = keras.layers.Conv2D(
            filters,
            1,
            1,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="query_conv2d",
        )
        self.key_conv2d = keras.layers.Conv2D(
            filters,
            1,
            1,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="key_conv2d",
        )
        self.value_conv2d = keras.layers.Conv2D(
            filters,
            1,
            1,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="value_conv2d",
        )
        self.softmax = keras.layers.Softmax(dtype="float32")
        self.output_conv2d = keras.layers.Conv2D(
            filters,
            1,
            1,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="output_conv2d",
        )

    def build(self, input_shape):
        self.group_norm.build(input_shape)
        self.query_conv2d.build(input_shape)
        self.key_conv2d.build(input_shape)
        self.value_conv2d.build(input_shape)
        self.output_conv2d.build(input_shape)

    def call(self, inputs, training=None):
        x = self.group_norm(inputs, training=training)
        query = self.query_conv2d(x, training=training)
        key = self.key_conv2d(x, training=training)
        value = self.value_conv2d(x, training=training)

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

        x = self.output_conv2d(x, training=training)
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


class ResNetBlock(keras.layers.Layer):
    """A ResNet block utilizing `GroupNormalization` and SiLU activation.

    Args:
        filters: The number of filters in the block.
        has_residual_projection: Whether to add a projection layer for the
            residual connection. Defaults to `False`.
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
        filters,
        has_residual_projection=False,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        self.filters = int(filters)
        self.has_residual_projection = bool(has_residual_projection)

        # === Layers ===
        self.norm1 = keras.layers.GroupNormalization(
            groups=32,
            axis=channel_axis,
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="norm1",
        )
        self.act1 = keras.layers.Activation("silu", dtype=self.dtype_policy)
        self.conv1 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=self.dtype_policy,
            name="conv1",
        )
        self.norm2 = keras.layers.GroupNormalization(
            groups=32,
            axis=channel_axis,
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="norm2",
        )
        self.act2 = keras.layers.Activation("silu", dtype=self.dtype_policy)
        self.conv2 = keras.layers.Conv2D(
            filters,
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=self.dtype_policy,
            name="conv2",
        )
        if self.has_residual_projection:
            self.residual_projection = keras.layers.Conv2D(
                filters,
                1,
                1,
                data_format=data_format,
                dtype=self.dtype_policy,
                name="residual_projection",
            )
        self.add = keras.layers.Add(dtype=self.dtype_policy)

    def build(self, input_shape):
        residual_shape = list(input_shape)
        self.norm1.build(input_shape)
        self.act1.build(input_shape)
        self.conv1.build(input_shape)
        input_shape = self.conv1.compute_output_shape(input_shape)
        self.norm2.build(input_shape)
        self.act2.build(input_shape)
        self.conv2.build(input_shape)
        input_shape = self.conv2.compute_output_shape(input_shape)
        if self.has_residual_projection:
            self.residual_projection.build(residual_shape)
        self.add.build([input_shape, input_shape])

    def call(self, inputs, training=None):
        x = inputs
        residual = x
        x = self.norm1(x, training=training)
        x = self.act1(x, training=training)
        x = self.conv1(x, training=training)
        x = self.norm2(x, training=training)
        x = self.act2(x, training=training)
        x = self.conv2(x, training=training)
        if self.has_residual_projection:
            residual = self.residual_projection(residual, training=training)
        x = self.add([residual, x])
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "has_residual_projection": self.has_residual_projection,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        outputs_shape = list(input_shape)
        if self.has_residual_projection:
            outputs_shape = self.residual_projection.compute_output_shape(
                outputs_shape
            )
        return outputs_shape


class VAEEncoder(keras.layers.Layer):
    """The encoder layer of VAE.

    Args:
        stackwise_num_filters: list of ints. The number of filters for each
            stack.
        stackwise_num_blocks: list of ints. The number of blocks for each stack.
        output_channels: int. The number of channels in the output. Defaults to
            `32`.
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
        stackwise_num_filters,
        stackwise_num_blocks,
        output_channels=32,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_num_blocks = stackwise_num_blocks
        self.output_channels = int(output_channels)
        self.data_format = data_format

        # === Layers ===
        self.input_projection = keras.layers.Conv2D(
            stackwise_num_filters[0],
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=self.dtype_policy,
            name="input_projection",
        )

        # Blocks.
        input_filters = stackwise_num_filters[0]
        self.blocks = []
        self.downsamples = []
        for i, filters in enumerate(stackwise_num_filters):
            for j in range(stackwise_num_blocks[i]):
                self.blocks.append(
                    ResNetBlock(
                        filters,
                        has_residual_projection=input_filters != filters,
                        data_format=data_format,
                        dtype=self.dtype_policy,
                        name=f"block_{i}_{j}",
                    )
                )
                input_filters = filters
            # No downsample in the last block.
            if i != len(stackwise_num_filters) - 1:
                self.downsamples.append(
                    keras.layers.ZeroPadding2D(
                        padding=((0, 1), (0, 1)),
                        data_format=data_format,
                        dtype=self.dtype_policy,
                        name=f"downsample_{i}_pad",
                    )
                )
                self.downsamples.append(
                    keras.layers.Conv2D(
                        filters,
                        3,
                        2,
                        data_format=data_format,
                        dtype=self.dtype_policy,
                        name=f"downsample_{i}_conv",
                    )
                )

        # Mid block.
        self.mid_block_0 = ResNetBlock(
            stackwise_num_filters[-1],
            has_residual_projection=False,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="mid_block_0",
        )
        self.mid_attention = Conv2DMultiHeadAttention(
            stackwise_num_filters[-1],
            data_format=data_format,
            dtype=self.dtype_policy,
            name="mid_attention",
        )
        self.mid_block_1 = ResNetBlock(
            stackwise_num_filters[-1],
            has_residual_projection=False,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="mid_block_1",
        )

        # Output layers.
        self.output_norm = keras.layers.GroupNormalization(
            groups=32,
            axis=channel_axis,
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="output_norm",
        )
        self.output_act = keras.layers.Activation(
            "swish", dtype=self.dtype_policy
        )
        self.output_projection = keras.layers.Conv2D(
            output_channels,
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=self.dtype_policy,
            name="output_projection",
        )

    def build(self, input_shape):
        self.input_projection.build(input_shape)
        input_shape = self.input_projection.compute_output_shape(input_shape)
        blocks_idx = 0
        downsamples_idx = 0
        for i, _ in enumerate(self.stackwise_num_filters):
            for _ in range(self.stackwise_num_blocks[i]):
                self.blocks[blocks_idx].build(input_shape)
                input_shape = self.blocks[blocks_idx].compute_output_shape(
                    input_shape
                )
                blocks_idx += 1
            if i != len(self.stackwise_num_filters) - 1:
                self.downsamples[downsamples_idx].build(input_shape)
                input_shape = self.downsamples[
                    downsamples_idx
                ].compute_output_shape(input_shape)
                downsamples_idx += 1
                self.downsamples[downsamples_idx].build(input_shape)
                input_shape = self.downsamples[
                    downsamples_idx
                ].compute_output_shape(input_shape)
                downsamples_idx += 1
        self.mid_block_0.build(input_shape)
        input_shape = self.mid_block_0.compute_output_shape(input_shape)
        self.mid_attention.build(input_shape)
        input_shape = self.mid_attention.compute_output_shape(input_shape)
        self.mid_block_1.build(input_shape)
        input_shape = self.mid_block_1.compute_output_shape(input_shape)
        self.output_norm.build(input_shape)
        self.output_act.build(input_shape)
        self.output_projection.build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        x = self.input_projection(x, training=training)
        blocks_idx = 0
        upsamples_idx = 0
        for i, _ in enumerate(self.stackwise_num_filters):
            for _ in range(self.stackwise_num_blocks[i]):
                x = self.blocks[blocks_idx](x, training=training)
                blocks_idx += 1
            if i != len(self.stackwise_num_filters) - 1:
                x = self.downsamples[upsamples_idx](x, training=training)
                x = self.downsamples[upsamples_idx + 1](x, training=training)
                upsamples_idx += 2
        x = self.mid_block_0(x, training=training)
        x = self.mid_attention(x, training=training)
        x = self.mid_block_1(x, training=training)
        x = self.output_norm(x, training=training)
        x = self.output_act(x, training=training)
        x = self.output_projection(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_num_blocks": self.stackwise_num_blocks,
                "output_channels": self.output_channels,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            h_axis, w_axis, c_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 1, 2, 3
        scale_factor = 2 ** (len(self.stackwise_num_filters) - 1)
        outputs_shape = list(input_shape)
        if (
            outputs_shape[h_axis] is not None
            and outputs_shape[w_axis] is not None
        ):
            outputs_shape[h_axis] = outputs_shape[h_axis] // scale_factor
            outputs_shape[w_axis] = outputs_shape[w_axis] // scale_factor
        outputs_shape[c_axis] = self.output_channels
        return outputs_shape


class VAEDecoder(keras.layers.Layer):
    """The decoder layer of VAE.

    Args:
        stackwise_num_filters: list of ints. The number of filters for each
            stack.
        stackwise_num_blocks: list of ints. The number of blocks for each stack.
        output_channels: int. The number of channels in the output. Defaults to
            `3`.
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
        stackwise_num_filters,
        stackwise_num_blocks,
        output_channels=3,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_num_blocks = stackwise_num_blocks
        self.output_channels = int(output_channels)
        self.data_format = data_format

        # === Layers ===
        self.input_projection = keras.layers.Conv2D(
            stackwise_num_filters[0],
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=self.dtype_policy,
            name="input_projection",
        )

        # Mid block.
        self.mid_block_0 = ResNetBlock(
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=self.dtype_policy,
            name="mid_block_0",
        )
        self.mid_attention = Conv2DMultiHeadAttention(
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=self.dtype_policy,
            name="mid_attention",
        )
        self.mid_block_1 = ResNetBlock(
            stackwise_num_filters[0],
            data_format=data_format,
            dtype=self.dtype_policy,
            name="mid_block_1",
        )

        # Blocks.
        input_filters = stackwise_num_filters[0]
        self.blocks = []
        self.upsamples = []
        for i, filters in enumerate(stackwise_num_filters):
            for j in range(stackwise_num_blocks[i]):
                self.blocks.append(
                    ResNetBlock(
                        filters,
                        has_residual_projection=input_filters != filters,
                        data_format=data_format,
                        dtype=self.dtype_policy,
                        name=f"block_{i}_{j}",
                    )
                )
                input_filters = filters
            # No upsample in the last block.
            if i != len(stackwise_num_filters) - 1:
                self.upsamples.append(
                    keras.layers.UpSampling2D(
                        2,
                        data_format=data_format,
                        dtype=self.dtype_policy,
                        name=f"upsample_{i}",
                    )
                )
                self.upsamples.append(
                    keras.layers.Conv2D(
                        filters,
                        3,
                        1,
                        padding="same",
                        data_format=data_format,
                        dtype=self.dtype_policy,
                        name=f"upsample_{i}_conv",
                    )
                )

        # Output layers.
        self.output_norm = keras.layers.GroupNormalization(
            groups=32,
            axis=channel_axis,
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="output_norm",
        )
        self.output_act = keras.layers.Activation(
            "swish", dtype=self.dtype_policy
        )
        self.output_projection = keras.layers.Conv2D(
            output_channels,
            3,
            1,
            padding="same",
            data_format=data_format,
            dtype=self.dtype_policy,
            name="output_projection",
        )

    def build(self, input_shape):
        self.input_projection.build(input_shape)
        input_shape = self.input_projection.compute_output_shape(input_shape)
        self.mid_block_0.build(input_shape)
        input_shape = self.mid_block_0.compute_output_shape(input_shape)
        self.mid_attention.build(input_shape)
        input_shape = self.mid_attention.compute_output_shape(input_shape)
        self.mid_block_1.build(input_shape)
        input_shape = self.mid_block_1.compute_output_shape(input_shape)
        blocks_idx = 0
        upsamples_idx = 0
        for i, _ in enumerate(self.stackwise_num_filters):
            for _ in range(self.stackwise_num_blocks[i]):
                self.blocks[blocks_idx].build(input_shape)
                input_shape = self.blocks[blocks_idx].compute_output_shape(
                    input_shape
                )
                blocks_idx += 1
            if i != len(self.stackwise_num_filters) - 1:
                self.upsamples[upsamples_idx].build(input_shape)
                input_shape = self.upsamples[
                    upsamples_idx
                ].compute_output_shape(input_shape)
                self.upsamples[upsamples_idx + 1].build(input_shape)
                input_shape = self.upsamples[
                    upsamples_idx + 1
                ].compute_output_shape(input_shape)
                upsamples_idx += 2
        self.output_norm.build(input_shape)
        self.output_act.build(input_shape)
        self.output_projection.build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        x = self.input_projection(x, training=training)
        x = self.mid_block_0(x, training=training)
        x = self.mid_attention(x, training=training)
        x = self.mid_block_1(x, training=training)
        blocks_idx = 0
        upsamples_idx = 0
        for i, _ in enumerate(self.stackwise_num_filters):
            for _ in range(self.stackwise_num_blocks[i]):
                x = self.blocks[blocks_idx](x, training=training)
                blocks_idx += 1
            if i != len(self.stackwise_num_filters) - 1:
                x = self.upsamples[upsamples_idx](x, training=training)
                x = self.upsamples[upsamples_idx + 1](x, training=training)
                upsamples_idx += 2
        x = self.output_norm(x, training=training)
        x = self.output_act(x, training=training)
        x = self.output_projection(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_num_blocks": self.stackwise_num_blocks,
                "output_channels": self.output_channels,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            h_axis, w_axis, c_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 1, 2, 3
        scale_factor = 2 ** (len(self.stackwise_num_filters) - 1)
        outputs_shape = list(input_shape)
        if (
            outputs_shape[h_axis] is not None
            and outputs_shape[w_axis] is not None
        ):
            outputs_shape[h_axis] = outputs_shape[h_axis] * scale_factor
            outputs_shape[w_axis] = outputs_shape[w_axis] * scale_factor
        outputs_shape[c_axis] = self.output_channels
        return outputs_shape


class DiagonalGaussianDistributionSampler(keras.layers.Layer):
    """A sampler for a diagonal Gaussian distribution.

    This layer samples latent variables from a diagonal Gaussian distribution.

    Args:
        method: str. The method used to sample from the distribution. Available
            methods are `"sample"` and `"mode"`. `"sample"` draws from the
            distribution using both the mean and log variance. `"mode"` draws
            from the distribution using the mean only.
        axis: int. The axis along which to split the mean and log variance.
            Defaults to `-1`.
        seed: optional int. Used as a random seed.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, method, axis=-1, seed=None, **kwargs):
        super().__init__(**kwargs)
        # TODO: Support `kl` and `nll` modes.
        valid_methods = ("sample", "mode")
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method {method}. Valid methods are "
                f"{list(valid_methods)}."
            )
        self.method = method
        self.axis = axis
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs):
        x = inputs
        if self.method == "sample":
            x_mean, x_logvar = ops.split(x, 2, axis=self.axis)
            x_logvar = ops.clip(x_logvar, -30.0, 20.0)
            x_std = ops.exp(ops.multiply(0.5, x_logvar))
            sample = keras.random.normal(
                ops.shape(x_mean), dtype=x_mean.dtype, seed=self.seed_generator
            )
            x = ops.add(x_mean, ops.multiply(x_std, sample))
        else:
            x, _ = ops.split(x, 2, axis=self.axis)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "seed": self.seed,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = output_shape[self.axis] // 2
        return output_shape
