import keras


BN_EPSILON = 1e-5
BN_MOMENTUM = 0.9
BN_AXIS = 3


class ConvBnActBlock(keras.layers.Layer):
    def __init__(
        self,
        filter, 
        activation, 
        name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        )
        self.bn = keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=f"{name}_bn",
        )
        self.act = keras.layers.Activation(activation)

    def build(self, input_shape):
        if self.name is None:
            self.name = keras.backend.get_uid("block0")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x

    def get_config(self):
        config = {
            "filter": self.filter,
            "activation": self.activation,
            "name": self.name,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
