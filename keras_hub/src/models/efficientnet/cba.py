import keras

BN_AXIS = 3


class CBABlock(keras.layers.Layer):
    """
    Args:
        input_filters: int, the number of input filters
        output_filters: int, the number of output filters
        kernel_size: default 3, the kernel_size to apply to the expansion phase
            convolutions
        strides: default 1, the strides to apply to the expansion phase
            convolutions
        data_format: str, channels_last (default) or channels_first, expects
            tensors to be of shape (N, H, W, C) or (N, C, H, W) respectively
        batch_norm_momentum: default 0.9, the BatchNormalization momentum
        batch_norm_epsilon: default 1e-3, the BatchNormalization epsilon
        activation: default "swish", the activation function used between
            convolution operations
        dropout: float, the optional dropout rate to apply before the output
            convolution, defaults to 0.2
        nores: bool, default False, forces no residual connection if True,
            otherwise allows it if False.

    Returns:
        A tensor representing a feature map, passed through the ConvBNAct
        block

    Note:
        Not intended to be used outside of the EfficientNet architecture.
    """

    def __init__(
        self,
        input_filters,
        output_filters,
        kernel_size=3,
        strides=1,
        data_format="channels_last",
        batch_norm_momentum=0.9,
        batch_norm_epsilon=1e-3,
        activation="swish",
        dropout=0.2,
        nores=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        self.activation = activation
        self.dropout = dropout
        self.nores = nores

        padding_pixels = kernel_size // 2
        self.conv1_pad = keras.layers.ZeroPadding2D(
            padding=(padding_pixels, padding_pixels),
            name=self.name + "conv_pad",
        )
        self.conv1 = keras.layers.Conv2D(
            filters=self.output_filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=self._conv_kernel_initializer(),
            padding="valid",
            data_format=data_format,
            use_bias=False,
            name=self.name + "conv",
        )
        self.bn1 = keras.layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.batch_norm_momentum,
            epsilon=self.batch_norm_epsilon,
            name=self.name + "bn",
        )
        self.act = keras.layers.Activation(
            self.activation, name=self.name + "activation"
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
        x = self.conv1_pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

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
        config = super().get_config()
        config.update(
            {
                "input_filters": self.input_filters,
                "output_filters": self.output_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "data_format": self.data_format,
                "batch_norm_momentum": self.batch_norm_momentum,
                "batch_norm_epsilon": self.batch_norm_epsilon,
                "activation": self.activation,
                "dropout": self.dropout,
                "nores": self.nores,
            }
        )

        return config
