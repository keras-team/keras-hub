import keras

from keras_hub.src.models.mobilenet.squeeze_and_excite_2d import (
    SqueezeAndExcite2D,
)
from keras_hub.src.models.mobilenet.util import adjust_channels

BN_EPSILON = 1e-5
BN_MOMENTUM = 0.9
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
        infilters,
        filters,
        kernel_size=3,
        stride=2,
        se=None,
        name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.infilters = infilters
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.se = se
        self.name = name

        channel_axis = (
            -1 if keras.config.image_data_format() == "channels_last" else 1
        )
        self.name = name = f"{name}_0"

        self.pad = keras.layers.ZeroPadding2D(
            padding=(1, 1),
            name=f"{name}_pad",
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
        )
        self.bn1 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn1",
        )
        self.act1 = keras.layers.ReLU()

        if se:
            self.se_layer = SqueezeAndExcite2D(
                filters=infilters,
                bottleneck_filters=adjust_channels(infilters * se),
                squeeze_activation="relu",
                excite_activation=keras.activations.hard_sigmoid,
                name=f"{name}_se",
            )

        self.conv2 = keras.layers.Conv2D(
            filters,
            kernel_size=1,
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=f"{name}_conv2",
        )
        self.bn2 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn2",
        )

    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")

    def call(self, inputs):
        x = self.pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.se_layer:
            x = self.se_layer(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return x

    def get_config(self):
        config = {
            "infilters": self.infilters,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "se": self.se,
            "name": self.name,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
