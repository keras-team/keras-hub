from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.ViTImageConverter")
class ViTImageConverter(ImageConverter):
    backbone_cls = ViTBackbone

    def __init__(
        self, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5], **kwargs
    ):
        super().__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    @preprocessing_function
    def call(self, inputs):
        x = super().call(inputs)
        # By default normalize using imagenet mean and std
        if self.norm_mean:
            x = x - self._expand_non_channel_dims(self.norm_mean, x)
        if self.norm_std:
            x = x / self._expand_non_channel_dims(self.norm_std, x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
            }
        )
        return config
