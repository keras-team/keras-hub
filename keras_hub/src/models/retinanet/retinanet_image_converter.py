import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.layers.RetinaNetImageConverter")
class RetinaNetImageConverter(ImageConverter):
    backbone_cls = RetinaNetBackbone

    def __init__(
        self,
        image_size=None,
        scale=None,
        offset=None,
        pad_to_aspect_ratio=False,
        crop_to_aspect_ratio=False,
        interpolation="bilinear",
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        data_format=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.scale = scale
        self.offset = offset
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.data_format = standardize_data_format(data_format)
        self.resizing = keras.layers.Resizing(
            height=image_size[0] if image_size else None,
            width=image_size[1] if image_size else None,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            interpolation=interpolation,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="resizing",
        )
        self.built = True

    def call(self, x, y=None):
        inputs = {
            "images": x,
        }
        if y is not None and isinstance(y, dict):
            for key in y:
                inputs[key] = y[key]
        # Resize images and bounding boxes
        inputs = self.resizing(inputs)
        x = inputs.pop("images")
        for key in inputs:
            y[key] = inputs[key]

        # Rescaling Image
        if self.scale is not None:
            x = x * self._expand_non_channel_dims(self.scale, x)
        if self.offset is not None:
            x = x + self._expand_non_channel_dims(self.offset, x)

        # By default normalize using imagenet mean and std
        if self.norm_mean:
            x = x - self._expand_non_channel_dims(self.norm_mean, x)

        if self.norm_std:
            x = x / self._expand_non_channel_dims(self.norm_std, x)

        return x, y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
                "pad_to_aspect_ratio": self.pad_to_aspect_ratio,
            }
        )
        return config
