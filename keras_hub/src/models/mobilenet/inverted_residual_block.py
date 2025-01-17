import keras

from keras_hub.src.models.mobilenet.squeeze_and_excite_2d import (
    SqueezeAndExcite2D,
)
from keras_hub.src.models.mobilenet.util import adjust_channels

BN_EPSILON = 1e-5
BN_MOMENTUM = 0.9
BN_AXIS = 3


class InvertedResidualBlock(keras.layers.Layer):
    """An Inverted Residual Block.

    Args:
        expansion: integer, the expansion ratio, multiplied with infilters to
            get the minimum value passed to adjust_channels.
        filters: integer, number of filters for convolution layer.
        kernel_size: integer, the kernel size for DepthWise Convolutions.
        stride: integer, the stride length for DepthWise Convolutions.
        se_ratio: float, ratio for bottleneck filters. Number of bottleneck
            filters = filters * se_ratio.
        activation: the activation layer to use.
        padding: padding in the conv2d layer
        name: string, block label.

    Returns:
        the updated input tensor.
    """

    def __init__(
        self,
        expansion,
        infilters,
        filters,
        kernel_size,
        stride,
        se_ratio,
        activation,
        padding,
        name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.expansion = expansion
        self.infilters = infilters
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.se_ratio = se_ratio
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
        )

        self.bn1 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn1",
        )

        self.act1 = keras.layers.Activation(activation=activation)

        self.pad = keras.layers.ZeroPadding2D(
            padding=(padding, padding),
            name=f"{name}_pad",
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
        )
        self.bn2 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn2",
        )

        self.act2 = keras.layers.Activation(activation=activation)

        self.se = None
        if self.se_ratio:
            se_filters = expanded_channels
            self.se = SqueezeAndExcite2D(
                filters=se_filters,
                bottleneck_filters=adjust_channels(se_filters * se_ratio),
                squeeze_activation="relu",
                excite_activation=keras.activations.hard_sigmoid,
                name=f"{name}_se",
            )

        self.conv3 = keras.layers.Conv2D(
            filters,
            kernel_size=1,
            data_format=keras.config.image_data_format(),
            use_bias=False,
            name=f"{name}_conv3",
        )
        self.bn3 = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn3",
        )

    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pad(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se:
            x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.stride == 1 and self.infilters == self.filters:
            x = inputs + x
        return x

    def get_config(self):
        config = {
            "expansion": self.expansion,
            "infilters": self.infilters,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "se_ratio": self.se_ratio,
            "activation": self.activation,
            "padding": self.padding,
            "name": self.name,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
