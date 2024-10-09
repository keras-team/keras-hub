from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.bounding_box.converters import convert_format
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.RetinaNetImageConverter")
class RetinaNetImageConverter(ImageConverter):
    backbone_cls = RetinaNetBackbone

    def __init__(
        self,
        ground_truth_bounding_box_format,
        target_bounding_box_format,
        image_size=None,
        scale=None,
        offset=None,
        crop_to_aspect_ratio=True,
        interpolation="bilinear",
        data_format=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ground_truth_bounding_box_format = ground_truth_bounding_box_format
        self.target_bounding_box_format = target_bounding_box_format
        self.image_size = image_size
        self.scale = scale
        self.offset = offset
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.interpolation = interpolation
        self.data_format = standardize_data_format(data_format)

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None, **kwargs):
        if self.image_size is not None:
            x = self.resizing(x)
        if self.offset is not None:
            x -= self._expand_non_channel_dims(self.offset, x)
        if self.scale is not None:
            x /= self._expand_non_channel_dims(self.scale, x)
        if y is not None and ops.is_tensor(y):
            y = convert_format(
                y,
                source=self.ground_truth_bounding_box_format,
                target=self.target_bounding_box_format,
                images=x,
            )
        return x, y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "ground_truth_bounding_box_format": self.ground_truth_bounding_box_format,
                "target_bounding_box_format": self.target_bounding_box_format,
            }
        )

        return config
