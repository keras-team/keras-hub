import keras

BN_AXIS = 3


class FusedMBConvBlock(keras.layers.Layer):
    """Implementation of the FusedMBConv block

    Also known as a Fused Mobile Inverted Residual Bottleneck block from:
    [EfficientNet-EdgeTPU](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
    [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298v3).

    FusedMBConv blocks are based on MBConv blocks, and replace the depthwise and
    1x1 output convolution blocks with a single 3x3 convolution block, fusing
    them together - hence the name "FusedMBConv". Alongside MBConv blocks, they
    can be used in mobile-oriented and efficient architectures, and are present
    in architectures EfficientNet.

    FusedMBConv blocks follow a narrow-wide-narrow structure - expanding a 1x1
    convolution, performing Squeeze-Excitation and then applying a 3x3
    convolution, which is a more efficient operation than conventional
    wide-narrow-wide structures.

    As they're frequently used for models to be deployed to edge devices,
    they're implemented as a layer for ease of use and re-use.

    Args:
        input_filters: int, the number of input filters
        output_filters: int, the number of output filters
        expand_ratio: default 1, the ratio by which input_filters are multiplied
            to expand the structure in the middle expansion phase
        kernel_size: default 3, the kernel_size to apply to the expansion phase
            convolutions
        strides: default 1, the strides to apply to the expansion phase
            convolutions
        data_format: str, channels_last (default) or channels_first, expects
            tensors to be of shape (N, H, W, C) or (N, C, H, W) respectively
        se_ratio: default 0.0, The filters used in the Squeeze-Excitation phase,
            and are chosen as the maximum between 1 and input_filters*se_ratio
        batch_norm_momentum: default 0.9, the BatchNormalization momentum
        batch_norm_epsilon: default 1e-3, float, epsilon for batch norm
            calcualtions. Used in denominator for calculations to prevent divide
            by 0 errors.
        activation: default "swish", the activation function used between
            convolution operations
        projection_activation: default None, the activation function to use
            after the output projection convoultion
        dropout: float, the optional dropout rate to apply before the output
            convolution, defaults to 0.2
        nores: bool, default False, forces no residual connection if True,
            otherwise allows it if False.
        projection_kernel_size: default 1, the kernel_size to apply to the
            output projection phase convolution

    Returns:
        A tensor representing a feature map, passed through the FusedMBConv
        block

    Note:
        Not intended to be used outside of the EfficientNet architecture.
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
