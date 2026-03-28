import keras
from keras import layers


class ZeroConv2D(layers.Layer):

    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv = layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )

    def call(self, inputs):
        return self.conv(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class ControlInjection(layers.Layer):

    def __init__(self, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.projection = ZeroConv2D(out_channels)

    def call(self, x, control):
        if x.shape[1:3] != control.shape[1:3]:
            raise ValueError(
                f"Spatial mismatch: {x.shape[1:3]} vs {control.shape[1:3]}"
            )
        control = self.projection(control)
        return x + control

    def get_config(self):
        config = super().get_config()
        config.update({"out_channels": self.out_channels})
        return config
