import keras

BN_AXIS = 3


class DepthwiseConvBlock(keras.layers.Layer):
    """
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    Args:
        x: Input tensor of shape `(rows, cols, channels)
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions. Specifying any stride value != 1 is
            incompatible with specifying any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    Input shape:
        4D tensor with shape: `(batch, rows, cols, channels)` in "channels_last"
        4D tensor with shape: `(batch, channels, rows, cols)` in "channels_first"
    Returns:
        Output tensor of block.
    """

    def __init__(
        self,
        input_filters,
        output_filters,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        data_format="channels_last",
        se_ratio=0.0,
        batch_norm_momentum=0.9,
        batch_norm_epsilon=1e-3,
        activation="swish",
        projection_activation=None,
        dropout=0.2,
        nores=False,
        projection_kernel_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.se_ratio = se_ratio
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        self.activation = activation
        self.projection_activation = projection_activation
        self.dropout = dropout
        self.nores = nores
        self.projection_kernel_size = projection_kernel_size
        self.filters = self.input_filters * self.expand_ratio
        self.filters_se = max(1, int(input_filters * se_ratio))

        padding_pixels = kernel_size // 2
        self.conv1_pad = keras.layers.ZeroPadding2D(
            padding=(padding_pixels, padding_pixels),
            name=self.name + "expand_conv_pad",
        )
        self.conv1 = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=self._conv_kernel_initializer(),
            padding="valid",
            data_format=data_format,
            use_bias=False,
            name=self.name + "expand_conv",
        )
        self.bn1 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon,
            name=self.name + "expand_bn",
        )
        self.act = keras.layers.Activation(
            self.activation, name=self.name + "expand_activation"
        )

        self.se_conv1 = keras.layers.Conv2D(
            self.filters_se,
            1,
            padding="same",
            data_format=data_format,
            activation=self.activation,
            kernel_initializer=self._conv_kernel_initializer(),
            name=self.name + "se_reduce",
        )

        self.se_conv2 = keras.layers.Conv2D(
            self.filters,
            1,
            padding="same",
            data_format=data_format,
            activation="sigmoid",
            kernel_initializer=self._conv_kernel_initializer(),
            name=self.name + "se_expand",
        )

        padding_pixels = projection_kernel_size // 2
        self.output_conv_pad = keras.layers.ZeroPadding2D(
            padding=(padding_pixels, padding_pixels),
            name=self.name + "project_conv_pad",
        )
        self.output_conv = keras.layers.Conv2D(
            filters=self.output_filters,
            kernel_size=projection_kernel_size,
            strides=1,
            kernel_initializer=self._conv_kernel_initializer(),
            padding="valid",
            data_format=data_format,
            use_bias=False,
            name=self.name + "project_conv",
        )

        self.bn2 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon,
            name=self.name + "project_bn",
        )

        if self.projection_activation:
            self.projection_act = keras.layers.Activation(
                self.projection_activation, name=self.name + "projection_act"
            )

        if self.dropout:
            self.dropout_layer = keras.layers.Dropout(
                self.dropout,
                noise_shape=(None, 1, 1, 1),
                name=self.name + "drop",
            )

    def _conv_kernel_initializer(
        self,
        scale=2.0,
        mode="fan_out",
        distribution="truncated_normal",
        seed=None,
    ):
        return keras.initializers.VarianceScaling(
            scale=scale, mode=mode, distribution=distribution, seed=seed
        )

    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")

    def call(self, inputs):
        # Expansion phase
        x = self.conv1_pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            se = keras.layers.GlobalAveragePooling2D(
                name=self.name + "se_squeeze",
                data_format=self.data_format,
            )(x)
            if BN_AXIS == 1:
                se_shape = (self.filters, 1, 1)
            else:
                se_shape = (1, 1, self.filters)

            se = keras.layers.Reshape(se_shape, name=self.name + "se_reshape")(
                se
            )

            se = self.se_conv1(se)
            se = self.se_conv2(se)

            x = keras.layers.multiply([x, se], name=self.name + "se_excite")

        # Output phase:
        x = self.output_conv_pad(x)
        x = self.output_conv(x)
        x = self.bn2(x)
        if self.expand_ratio == 1 and self.projection_activation:
            x = self.projection_act(x)

        # Residual:
        if (
            self.strides == 1
            and self.input_filters == self.output_filters
            and not self.nores
        ):
            if self.dropout:
                x = self.dropout_layer(x)
            x = keras.layers.Add(name=self.name + "add")([x, inputs])
        return x

    def get_config(self):
        config = {
            "input_filters": self.input_filters,
            "output_filters": self.output_filters,
            "expand_ratio": self.expand_ratio,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "data_format": self.data_format,
            "se_ratio": self.se_ratio,
            "batch_norm_momentum": self.batch_norm_momentum,
            "batch_norm_epsilon": self.batch_norm_epsilon,
            "activation": self.activation,
            "projection_activation": self.projection_activation,
            "dropout": self.dropout,
            "nores": self.nores,
            "projection_kernel_size": self.projection_kernel_size,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def apply_depthwise_conv_block(
    x, filters, kernel_size=3, stride=2, se=None, name=None
):
    """Adds a depthwise convolution block.

    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    Args:
        x: Input tensor of shape `(rows, cols, channels)
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions. Specifying any stride value != 1 is
            incompatible with specifying any `dilation_rate` value != 1.
        block_id: Integer, a unique identification designating the block number.

    Input shape:
        4D tensor with shape: `(batch, rows, cols, channels)` in "channels_last"
        4D tensor with shape: `(batch, channels, rows, cols)` in "channels_first"
    Returns:
        Output tensor of block.
    """
    channel_axis = (
        -1 if keras.config.image_data_format() == "channels_last" else 1
    )
    infilters = x.shape[channel_axis]
    name = f"{name}_0"

    x = keras.layers.ZeroPadding2D(
        padding=(1, 1),
        name=f"{name}_pad",
    )(x)
    x = keras.layers.Conv2D(
        infilters,
        kernel_size,
        strides=stride,
        padding="valid",
        data_format=keras.config.image_data_format(),
        groups=infilters,
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=channel_axis,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=f"{name}_bn1",
    )(x)
    x = keras.layers.ReLU()(x)

    if se:
        x = SqueezeAndExcite2D(
            input=x,
            filters=infilters,
            bottleneck_filters=adjust_channels(infilters * se),
            squeeze_activation="relu",
            excite_activation=keras.activations.hard_sigmoid,
            name=f"{name}_se",
        )

    x = keras.layers.Conv2D(
        filters,
        kernel_size=1,
        data_format=keras.config.image_data_format(),
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=channel_axis,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=f"{name}_bn2",
    )(x)
    return x