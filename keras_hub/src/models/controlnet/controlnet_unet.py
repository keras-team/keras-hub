import keras
from .controlnet_layers import ControlInjection


class ControlNetUNet(keras.Model):

    def __init__(self, base_channels=64, **kwargs):
        super().__init__(**kwargs)

        self.base_channels = base_channels

        self.conv1 = keras.layers.Conv2D(
            base_channels, 3, padding="same", activation="relu"
        )

        self.inject = ControlInjection(base_channels)

        self.conv2 = keras.layers.Conv2D(
            base_channels, 3, padding="same", activation="relu"
        )

        self.out_conv = keras.layers.Conv2D(
            3, 1, padding="same"
        )

    def call(self, image, control_features):
        if "scale_1" not in control_features:
            raise ValueError("Expected 'scale_1' in control_features.")

        x = self.conv1(image)
        x = self.inject(x, control_features["scale_1"])
        x = self.conv2(x)
        x = self.out_conv(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"base_channels": self.base_channels})
        return config
