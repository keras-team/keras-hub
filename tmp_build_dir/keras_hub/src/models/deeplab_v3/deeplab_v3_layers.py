import keras
from keras import ops


class SpatialPyramidPooling(keras.layers.Layer):
    """Implements the Atrous Spatial Pyramid Pooling.

    Reference for Atrous Spatial Pyramid Pooling [Rethinking Atrous Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf) and
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

    Args:
    dilation_rates: list of ints. The dilation rate for parallel dilated conv.
        Usually a sample choice of rates are `[6, 12, 18]`.
    num_channels: int. The number of output channels, defaults to `256`.
    activation: str. Activation to be used, defaults to `relu`.
    dropout: float. The dropout rate of the final projection output after the
        activations and batch norm, defaults to `0.0`, which means no dropout is
        applied to the output.

    Example:
    ```python
    inp = keras.layers.Input((384, 384, 3))
    backbone = keras.applications.EfficientNetB0(
        input_tensor=inp,
        include_top=False)
    output = backbone(inp)
    output = SpatialPyramidPooling(
        dilation_rates=[6, 12, 18])(output)
    ```
    """

    def __init__(
        self,
        dilation_rates,
        num_channels=256,
        activation="relu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dilation_rates = dilation_rates
        self.num_channels = num_channels
        self.activation = activation
        self.dropout = dropout
        self.data_format = keras.config.image_data_format()
        self.channel_axis = -1 if self.data_format == "channels_last" else 1

    def build(self, input_shape):
        channels = input_shape[self.channel_axis]

        # This is the parallel networks that process the input features with
        # different dilation rates. The output from each channel will be merged
        # together and feed to the output.
        self.aspp_parallel_channels = []

        # Channel1 with Conv2D and 1x1 kernel size.
        conv_sequential = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                    data_format=self.data_format,
                    name="aspp_conv_1",
                ),
                keras.layers.BatchNormalization(
                    axis=self.channel_axis, name="aspp_bn_1"
                ),
                keras.layers.Activation(
                    self.activation, name="aspp_activation_1"
                ),
            ]
        )
        conv_sequential.build(input_shape)
        self.aspp_parallel_channels.append(conv_sequential)

        # Channel 2 and afterwards are based on self.dilation_rates, and each of
        # them will have conv2D with 3x3 kernel size.
        for i, dilation_rate in enumerate(self.dilation_rates):
            conv_sequential = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        filters=self.num_channels,
                        kernel_size=(3, 3),
                        padding="same",
                        dilation_rate=dilation_rate,
                        use_bias=False,
                        data_format=self.data_format,
                        name=f"aspp_conv_{i + 2}",
                    ),
                    keras.layers.BatchNormalization(
                        axis=self.channel_axis, name=f"aspp_bn_{i + 2}"
                    ),
                    keras.layers.Activation(
                        self.activation, name=f"aspp_activation_{i + 2}"
                    ),
                ]
            )
            conv_sequential.build(input_shape)
            self.aspp_parallel_channels.append(conv_sequential)

        # Last channel is the global average pooling with conv2D 1x1 kernel.
        if self.channel_axis == -1:
            reshape = keras.layers.Reshape((1, 1, channels), name="reshape")
        else:
            reshape = keras.layers.Reshape((channels, 1, 1), name="reshape")
        pool_sequential = keras.Sequential(
            [
                keras.layers.GlobalAveragePooling2D(
                    data_format=self.data_format, name="average_pooling"
                ),
                reshape,
                keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                    data_format=self.data_format,
                    name="conv_pooling",
                ),
                keras.layers.BatchNormalization(
                    axis=self.channel_axis, name="bn_pooling"
                ),
                keras.layers.Activation(
                    self.activation, name="activation_pooling"
                ),
            ]
        )
        pool_sequential.build(input_shape)
        self.aspp_parallel_channels.append(pool_sequential)

        # Final projection layers
        projection = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                    data_format=self.data_format,
                    name="conv_projection",
                ),
                keras.layers.BatchNormalization(
                    axis=self.channel_axis, name="bn_projection"
                ),
                keras.layers.Activation(
                    self.activation, name="activation_projection"
                ),
                keras.layers.Dropout(rate=self.dropout, name="dropout"),
            ],
        )
        projection_input_channels = (
            2 + len(self.dilation_rates)
        ) * self.num_channels
        if self.data_format == "channels_first":
            projection.build(
                (input_shape[0],)
                + (projection_input_channels,)
                + (input_shape[2:])
            )
        else:
            projection.build((input_shape[:-1]) + (projection_input_channels,))
        self.projection = projection
        self.built = True

    def call(self, inputs):
        """Calls the Atrous Spatial Pyramid Pooling layer on an input.

        Args:
            inputs: A tensor of shape [batch, height, width, channels]

        Returns:
            A tensor of shape [batch, height, width, num_channels]
        """
        result = []

        for channel in self.aspp_parallel_channels:
            temp = ops.cast(channel(inputs), inputs.dtype)
            result.append(temp)

        image_shape = ops.shape(inputs)
        if self.channel_axis == -1:
            height, width = image_shape[1], image_shape[2]
        else:
            height, width = image_shape[2], image_shape[3]
        result[-1] = keras.layers.Resizing(
            height,
            width,
            interpolation="bilinear",
            data_format=self.data_format,
            name="resizing",
        )(result[-1])

        result = ops.concatenate(result, axis=self.channel_axis)
        return self.projection(result)

    def compute_output_shape(self, inputs_shape):
        if self.data_format == "channels_first":
            return tuple(
                (inputs_shape[0],) + (self.num_channels,) + (inputs_shape[2:])
            )
        else:
            return tuple((inputs_shape[:-1]) + (self.num_channels,))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dilation_rates": self.dilation_rates,
                "num_channels": self.num_channels,
                "activation": self.activation,
                "dropout": self.dropout,
            }
        )
        return config
