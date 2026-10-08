import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mobilenet.util import adjust_channels

BN_EPSILON = 1e-5
BN_MOMENTUM = 0.9


class SqueezeAndExcite2D(keras.layers.Layer):
    """
    Description:
        This layer applies a content-aware mechanism to adaptively assign
        channel-wise weights. It uses global average pooling to compress
        feature maps into single values, which are then processed by
        two Conv1D layers: the first reduces the dimensionality, and
        the second restores it.
    Args:
        filters: Number of input and output filters. The number of input and
            output filters is same.
        bottleneck_filters: (Optional) Number of bottleneck filters. Defaults
            to `0.25 * filters`
        squeeze_activation: (Optional) String, callable (or
            keras.layers.Layer) or keras.activations.Activation instance
            denoting activation to be applied after squeeze convolution.
            Defaults to `relu`.
        excite_activation: (Optional) String, callable (or
            keras.layers.Layer) or keras.activations.Activation instance
            denoting activation to be applied after excite convolution.
            Defaults to `sigmoid`.
        name: Name of the layer
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.
    """

    def __init__(
        self,
        filters,
        bottleneck_filters=None,
        squeeze_activation="relu",
        excite_activation="sigmoid",
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filters = filters
        self.bottleneck_filters = bottleneck_filters
        self.squeeze_activation = squeeze_activation
        self.excite_activation = excite_activation
        self.name = name

        image_data_format = keras.config.image_data_format()
        if image_data_format == "channels_last":
            self.spatial_dims = (1, 2)
        else:
            self.spatial_dims = (2, 3)

        self.conv_reduce = keras.layers.Conv2D(
            bottleneck_filters,
            (1, 1),
            data_format=image_data_format,
            name=f"{name}_conv_reduce",
            dtype=dtype,
        )
        self.activation1 = keras.layers.Activation(
            self.squeeze_activation,
            name=self.name + "squeeze_activation",
            dtype=dtype,
        )

        self.conv_expand = keras.layers.Conv2D(
            filters,
            (1, 1),
            data_format=image_data_format,
            name=f"{name}_conv_expand",
            dtype=dtype,
        )
        self.gate = keras.layers.Activation(
            self.excite_activation,
            name=self.name + "excite_activation",
            dtype=dtype,
        )

    def compute_output_shape(self, input_shape):
        shape = self.conv_reduce.compute_output_shape(input_shape)
        shape = self.activation1.compute_output_shape(shape)
        shape = self.conv_expand.compute_output_shape(shape)
        return self.gate.compute_output_shape(shape)

    def build(self, input_shape):
        self.conv_reduce.build(input_shape)
        input_shape = self.conv_reduce.compute_output_shape(input_shape)
        self.activation1.build(input_shape)
        input_shape = self.activation1.compute_output_shape(input_shape)
        self.conv_expand.build(input_shape)
        input_shape = self.conv_expand.compute_output_shape(input_shape)
        self.gate.build(input_shape)
        self.built = True

    def call(self, inputs):
        x_se = keras.ops.mean(inputs, axis=self.spatial_dims, keepdims=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.activation1(x_se)
        x_se = self.conv_expand(x_se)
        return inputs * self.gate(x_se)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "bottleneck_filters": self.bottleneck_filters,
                "squeeze_activation": self.squeeze_activation,
                "excite_activation": self.excite_activation,
                "name": self.name,
                "spatial_dims": self.spatial_dims,
            }
        )
        return config


class DepthwiseConvBlock(keras.layers.Layer):
    """
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu, optional squeeze & excite, pointwise convolution,
    and batch normalization layer.

    Args:
        infilters: int, the output channels for the initial depthwise conv
        filters: int, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        kernel_size: int or Tuple[int, int], the kernel size to apply
            to the initial depthwise convolution
        strides: An int or Tuple[int, int], specifying the strides
            of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        squeeze_excite_ratio: squeeze & excite ratio: float[Optional], if
            exists, specifies the ratio of channels (<1) to squeeze the initial
            signal into before reexciting back out. If (>1) technically, it's an
            excite & squeeze layer. If this doesn't exist there is no
            SqueezeExcite layer.
        residual: bool, default False. True if we want a residual connection. If
            False, there is no residual connection.
        name: str, name of the layer
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Input shape when applied as a layer:
        4D tensor with shape: `(batch, rows, cols, channels)` in "channels_last"
        4D tensor with shape: `(batch, channels, rows, cols)` in
            "channels_first"
    Returns:
        Output tensor of block.
    """

    def __init__(
        self,
        infilters,
        filters,
        kernel_size=3,
        stride=2,
        squeeze_excite_ratio=None,
        residual=False,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.infilters = infilters
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.squeeze_excite_ratio = squeeze_excite_ratio
        self.residual = residual
        self.name = name

        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )
        self.name = name = f"{name}_0"

        self.pad = keras.layers.ZeroPadding2D(
            padding=(1, 1),
            name=f"{name}_pad",
            dtype=dtype,
        )
        self.conv1 = keras.layers.Conv2D(
            infilters,
            kernel_size,
            strides=stride,
            padding="valid",
            data_format=keras.config.image_data_format(),
            groups=infilters,
            use_bias=False,
            name=f"{name}_conv1",
            dtype=dtype,
        )
        self.batch_normalization1 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn1",
            dtype=dtype,
        )
        self.activation1 = keras.layers.ReLU(dtype=dtype)

        if squeeze_excite_ratio:
            self.se_layer = SqueezeAndExcite2D(
                filters=infilters,
                bottleneck_filters=adjust_channels(
                    infilters * squeeze_excite_ratio
                ),
                squeeze_activation="relu",
                excite_activation=keras.activations.hard_sigmoid,
                name=f"{name}_squeeze_excite",
                dtype=dtype,
            )

        self.conv2 = keras.layers.Conv2D(
            filters,
            kernel_size=1,
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=f"{name}_conv2",
            dtype=dtype,
        )
        self.batch_normalization2 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn2",
            dtype=dtype,
        )

    def build(self, input_shape):
        self.pad.build(input_shape)
        input_shape = self.pad.compute_output_shape(input_shape)
        self.conv1.build(input_shape)
        input_shape = self.conv1.compute_output_shape(input_shape)
        self.batch_normalization1.build(input_shape)
        input_shape = self.batch_normalization1.compute_output_shape(
            input_shape
        )
        self.activation1.build(input_shape)
        input_shape = self.activation1.compute_output_shape(input_shape)
        if self.squeeze_excite_ratio:
            self.se_layer.build(input_shape)
            input_shape = self.se_layer.compute_output_shape(input_shape)
        self.conv2.build(input_shape)
        input_shape = self.conv2.compute_output_shape(input_shape)
        self.batch_normalization2.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.pad(inputs)
        x = self.conv1(x)
        x = self.batch_normalization1(x)
        x = self.activation1(x)

        if self.squeeze_excite_ratio:
            x = self.se_layer(x)

        x = self.conv2(x)
        x = self.batch_normalization2(x)

        if self.residual:
            x = x + inputs

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "infilters": self.infilters,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "squeeze_excite_ratio": self.squeeze_excite_ratio,
                "residual": self.residual,
                "name": self.name,
            }
        )
        return config


class InvertedResidualBlock(keras.layers.Layer):
    """An Inverted Residual Block.

    Args:
        expansion: integer, the expansion ratio, multiplied with infilters to
            get the minimum value passed to adjust_channels.
        infilters: Int, the output channels for the initial depthwise conv
        filters: integer, number of filters for convolution layer.
        kernel_size: integer, the kernel size for DepthWise Convolutions.
        stride: integer, the stride length for DepthWise Convolutions.
        squeeze_excite_ratio: float, ratio for bottleneck filters. Number of
            bottleneck filters = filters * se_ratio.
        activation: the activation layer to use.
        padding: padding in the conv2d layer
        name: string, block label.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Input shape when applied as a layer:
        4D tensor with shape: `(batch, rows, cols, channels)` in "channels_last"
        4D tensor with shape: `(batch, channels, rows, cols)` in
            "channels_first"
    Returns:
        Output tensor of block.
    """

    def __init__(
        self,
        expansion,
        infilters,
        filters,
        kernel_size,
        stride,
        squeeze_excite_ratio,
        activation,
        padding,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.expansion = expansion
        self.infilters = infilters
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.squeeze_excite_ratio = squeeze_excite_ratio
        self.activation = activation
        self.padding = padding
        self.name = name

        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )
        expanded_channels = adjust_channels(expansion)

        self.conv1 = keras.layers.Conv2D(
            expanded_channels,
            kernel_size=1,
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=f"{name}_conv1",
            dtype=dtype,
        )

        self.batch_normalization1 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn1",
            dtype=dtype,
        )

        self.activation1 = keras.layers.Activation(
            activation=activation,
            dtype=dtype,
        )

        self.pad = keras.layers.ZeroPadding2D(
            padding=(padding, padding),
            name=f"{name}_pad",
            dtype=dtype,
        )

        self.conv2 = keras.layers.Conv2D(
            expanded_channels,
            kernel_size,
            strides=stride,
            padding="valid",
            groups=expanded_channels,
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=f"{name}_conv2",
            dtype=dtype,
        )
        self.batch_normalization2 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn2",
            dtype=dtype,
        )

        self.activation2 = keras.layers.Activation(
            activation=activation,
            dtype=dtype,
        )

        self.squeeze_excite = None
        if self.squeeze_excite_ratio:
            se_filters = expanded_channels
            self.squeeze_excite = SqueezeAndExcite2D(
                filters=se_filters,
                bottleneck_filters=adjust_channels(
                    se_filters * squeeze_excite_ratio
                ),
                squeeze_activation="relu",
                excite_activation=keras.activations.hard_sigmoid,
                name=f"{name}_se",
                dtype=dtype,
            )

        self.conv3 = keras.layers.Conv2D(
            filters,
            kernel_size=1,
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=f"{name}_conv3",
            dtype=dtype,
        )
        self.batch_normalization3 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn3",
            dtype=dtype,
        )

    def build(self, input_shape):
        self.conv1.build(input_shape)
        input_shape = self.conv1.compute_output_shape(input_shape)
        self.batch_normalization1.build(input_shape)
        input_shape = self.batch_normalization1.compute_output_shape(
            input_shape
        )
        self.activation1.build(input_shape)
        input_shape = self.activation1.compute_output_shape(input_shape)
        self.pad.build(input_shape)
        input_shape = self.pad.compute_output_shape(input_shape)
        self.conv2.build(input_shape)
        input_shape = self.conv2.compute_output_shape(input_shape)
        self.batch_normalization2.build(input_shape)
        input_shape = self.batch_normalization2.compute_output_shape(
            input_shape
        )
        self.activation2.build(input_shape)
        input_shape = self.activation2.compute_output_shape(input_shape)
        if self.squeeze_excite_ratio:
            self.squeeze_excite.build(input_shape)
            input_shape = self.squeeze_excite.compute_output_shape(input_shape)
        self.conv3.build(input_shape)
        input_shape = self.conv3.compute_output_shape(input_shape)
        self.batch_normalization3.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_normalization1(x)
        x = self.activation1(x)
        x = self.pad(x)
        x = self.conv2(x)
        x = self.batch_normalization2(x)
        x = self.activation2(x)
        if self.squeeze_excite:
            x = self.squeeze_excite(x)
        x = self.conv3(x)
        x = self.batch_normalization3(x)
        if self.stride == 1 and self.infilters == self.filters:
            x = inputs + x
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "expansion": self.expansion,
                "infilters": self.infilters,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "squeeze_excite_ratio": self.squeeze_excite_ratio,
                "activation": self.activation,
                "padding": self.padding,
                "name": self.name,
            }
        )
        return config


class ConvBnActBlock(keras.layers.Layer):
    """
    A ConvBnActBlock consists of a convultion, batchnorm, and activation layer

    Args:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        activation: The activation function to apply to the signal at the end.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Input shape (when called as a layer):
        4D tensor with shape: `(batch, rows, cols, channels)` in "channels_last"
        4D tensor with shape: `(batch, channels, rows, cols)` in
            "channels_first"

    Returns:
        Output tensor of block.
    """

    def __init__(
        self,
        filter,
        activation,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.filter = filter
        self.activation = activation
        self.name = name

        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )
        self.conv = keras.layers.Conv2D(
            filter,
            kernel_size=1,
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=f"{name}_conv",
            dtype=dtype,
        )
        self.batch_normalization = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn",
            dtype=dtype,
        )
        self.activation_layer = keras.layers.Activation(
            activation,
            dtype=dtype,
        )

    def build(self, input_shape):
        self.conv.build(input_shape)
        input_shape = self.conv.compute_output_shape(input_shape)
        self.batch_normalization.build(input_shape)
        input_shape = self.batch_normalization.compute_output_shape(input_shape)
        self.activation_layer.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_normalization(x)
        x = self.activation_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter": self.filter,
                "activation": self.activation,
                "name": self.name,
            }
        )
        return config


@keras_hub_export("keras_hub.models.MobileNetBackbone")
class MobileNetBackbone(Backbone):
    """Instantiates the MobileNet architecture.

    MobileNet is a lightweight convolutional neural network (CNN)
    optimized for mobile and edge devices, striking a balance between
    accuracy and efficiency. By employing depthwise separable convolutions
    and techniques like Squeeze-and-Excitation (SE) blocks,
    MobileNet models are highly suitable for real-time applications on
    resource-constrained devices.

    References:
        - [MobileNets: Efficient Convolutional Neural Networks
       for Mobile Vision Applications](
        https://arxiv.org/abs/1704.04861)
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)
        - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
        (ICCV 2019)

    Args:
        stackwise_expansion: list of list of ints, the expanded filters for
            each inverted residual block for each block in the model.
        stackwise_num_blocks: list of ints, number of inversted residual blocks
            per block
        stackwise_num_filters: list of list of ints, number of filters for
            each inverted residual block in the model.
        stackwise_kernel_size: list of list of ints, kernel size for each
            inverted residual block in the model.
        stackwise_num_strides: list of list of ints, stride length for each
            inverted residual block in the model.
        stackwise_se_ratio: se ratio for each inverted residual block in the
            model. 0 if dont want to add Squeeze and Excite layer.
        stackwise_activation: list of list of activation functions, for each
            inverted residual block in the model.
        stackwise_padding: list of list of int, to provide padding values for
            each inverted residual block in the model.
        output_num_filters: specifies whether to add conv and batch_norm in the
            end, if set to None, it will not add these layers in the end.
            'None' for MobileNetV1
        depthwise_filters: int, number of filters in depthwise separable
            convolution layer,
        last_layer_filter: int, channels/filters for the head ConvBnAct block
        squeeze_and_excite: float, squeeze and excite ratio in the depthwise
            layer, None, if dont want to do squeeze and excite
        image_shape: optional shape tuple, defaults to (224, 224, 3).
        input_activation: activation function to be used in the input layer
            'hard_swish' for MobileNetV3,
            'relu6' for MobileNetV1 and MobileNetV2
        output_activation: activation function to be used in the output layer
            'hard_swish' for MobileNetV3,
            'relu6' for MobileNetV1 and MobileNetV2
        input_num_filters: int, channels/filters for the input before the stem
            input_conv
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.


    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = MobileNetBackbone(
        stackwise_expansion=[
                [40, 56],
                [64, 144, 144],
                [72, 72],
                [144, 288, 288],
            ],
            stackwise_num_blocks=[2, 3, 2, 3],
            stackwise_num_filters=[
                [16, 16],
                [24, 24, 24],
                [24, 24],
                [48, 48, 48],
            ],
            stackwise_kernel_size=[[3, 3], [5, 5, 5], [5, 5], [5, 5, 5]],
            stackwise_num_strides=[[2, 1], [2, 1, 1], [1, 1], [2, 1, 1]],
            stackwise_se_ratio=[
                [None, None],
                [0.25, 0.25, 0.25],
                [0.3, 0.3],
                [0.3, 0.25, 0.25],
            ],
            stackwise_activation=[
                ["relu", "relu"],
                ["hard_swish", "hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish"],
                ["hard_swish", "hard_swish", "hard_swish"],
            ],
            output_num_filters=288,
            input_activation="hard_swish",
            output_activation="hard_swish",
            input_num_filters=16,
            image_shape=(224, 224, 3),
            depthwise_filters=8,
            squeeze_and_excite=0.5,

    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_expansion,
        stackwise_num_blocks,
        stackwise_num_filters,
        stackwise_kernel_size,
        stackwise_num_strides,
        stackwise_se_ratio,
        stackwise_activation,
        stackwise_padding,
        output_num_filters,
        depthwise_filters,
        depthwise_stride,
        depthwise_residual,
        last_layer_filter,
        squeeze_and_excite=None,
        image_shape=(None, None, 3),
        input_activation="hard_swish",
        output_activation="hard_swish",
        input_num_filters=16,
        dtype=None,
        **kwargs,
    ):
        # === Functional Model ===
        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )

        image_input = keras.layers.Input(shape=image_shape)
        x = image_input
        input_num_filters = adjust_channels(input_num_filters)

        x = keras.layers.ZeroPadding2D(
            padding=(1, 1),
            name="input_pad",
            dtype=dtype,
        )(x)
        x = keras.layers.Conv2D(
            input_num_filters,
            kernel_size=3,
            strides=(2, 2),
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name="input_conv",
            dtype=dtype,
        )(x)
        x = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="input_batch_norm",
            dtype=dtype,
        )(x)
        x = keras.layers.Activation(
            input_activation,
            dtype=dtype,
        )(x)

        x = DepthwiseConvBlock(
            input_num_filters,
            depthwise_filters,
            stride=depthwise_stride,
            squeeze_excite_ratio=squeeze_and_excite,
            residual=depthwise_residual,
            name="block_0",
            dtype=dtype,
        )(x)

        for block in range(len(stackwise_num_blocks)):
            for inverted_block in range(stackwise_num_blocks[block]):
                infilters = x.shape[channel_axis]
                x = InvertedResidualBlock(
                    expansion=stackwise_expansion[block][inverted_block],
                    infilters=infilters,
                    filters=adjust_channels(
                        stackwise_num_filters[block][inverted_block]
                    ),
                    kernel_size=stackwise_kernel_size[block][inverted_block],
                    stride=stackwise_num_strides[block][inverted_block],
                    squeeze_excite_ratio=stackwise_se_ratio[block][
                        inverted_block
                    ],
                    activation=stackwise_activation[block][inverted_block],
                    padding=stackwise_padding[block][inverted_block],
                    name=f"block_{block + 1}_{inverted_block}",
                    dtype=dtype,
                )(x)

        x = ConvBnActBlock(
            filter=adjust_channels(last_layer_filter),
            activation="hard_swish",
            name=f"block_{len(stackwise_num_blocks) + 1}_0",
            dtype=dtype,
        )(x)

        super().__init__(inputs=image_input, outputs=x, dtype=dtype, **kwargs)

        # === Config ===
        self.stackwise_expansion = stackwise_expansion
        self.stackwise_num_blocks = stackwise_num_blocks
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_kernel_size = stackwise_kernel_size
        self.stackwise_num_strides = stackwise_num_strides
        self.stackwise_se_ratio = stackwise_se_ratio
        self.stackwise_activation = stackwise_activation
        self.stackwise_padding = stackwise_padding
        self.input_num_filters = input_num_filters
        self.output_num_filters = output_num_filters
        self.depthwise_filters = depthwise_filters
        self.depthwise_stride = depthwise_stride
        self.depthwise_residual = depthwise_residual
        self.last_layer_filter = last_layer_filter
        self.squeeze_and_excite = squeeze_and_excite
        self.input_activation = input_activation
        self.output_activation = output_activation
        self.image_shape = image_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_expansion": self.stackwise_expansion,
                "stackwise_num_blocks": self.stackwise_num_blocks,
                "stackwise_num_filters": self.stackwise_num_filters,
                "stackwise_kernel_size": self.stackwise_kernel_size,
                "stackwise_num_strides": self.stackwise_num_strides,
                "stackwise_se_ratio": self.stackwise_se_ratio,
                "stackwise_activation": self.stackwise_activation,
                "stackwise_padding": self.stackwise_padding,
                "image_shape": self.image_shape,
                "input_num_filters": self.input_num_filters,
                "output_num_filters": self.output_num_filters,
                "depthwise_filters": self.depthwise_filters,
                "depthwise_stride": self.depthwise_stride,
                "depthwise_residual": self.depthwise_residual,
                "last_layer_filter": self.last_layer_filter,
                "squeeze_and_excite": self.squeeze_and_excite,
                "input_activation": self.input_activation,
                "output_activation": self.output_activation,
            }
        )
        return config
