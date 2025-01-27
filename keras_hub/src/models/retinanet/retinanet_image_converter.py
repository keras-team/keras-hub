import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.RetinaNetImageConverter")
class RetinaNetImageConverter(ImageConverter):
    backbone_cls = RetinaNetBackbone

    def __init__(
        self,
        image_size=None,
        scale=None,
        offset=None,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        bounding_box_format="yxyx",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resizing = keras.layers.Resizing(
            height=self.image_size[0] if image_size else None,
            width=self.image_size[1] if image_size else None,
            bounding_box_format=bounding_box_format,
            crop_to_aspect_ratio=self.crop_to_aspect_ratio,
            pad_to_aspect_ratio=self.pad_to_aspect_ratio,
            interpolation=self.interpolation,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="resizing",
        )

        self.image_size = image_size
        self.scale = scale
        self.offset = offset
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.bounding_box_format = bounding_box_format

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        if y is not None:
            inputs = self.resizing({"images": x, "bounding_boxes": y})
            x = inputs["images"]
            y = inputs["bounding_boxes"]
        else:
            x = self.resizing(x)
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

        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config
