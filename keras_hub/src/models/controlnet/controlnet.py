import keras

from .controlnet_backbone import ControlNetBackbone
from .controlnet_preprocessor import ControlNetPreprocessor
from .controlnet_unet import ControlNetUNet


class ControlNet(keras.Model):

    def __init__(self, image_size=128, base_channels=64, **kwargs):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.base_channels = base_channels

        self.preprocessor = ControlNetPreprocessor(
            target_size=(image_size, image_size)
        )
        self.backbone = ControlNetBackbone()
        self.unet = ControlNetUNet(base_channels=base_channels)

    def call(self, inputs):
        image = inputs["image"]
        control = inputs["control"]
        
        image = self.preprocessor(image)
        control = self.preprocessor(control)
        control_features = self.backbone(control)

        output = self.unet(image, control_features)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "base_channels": self.base_channels,
            }
        )
        return config
