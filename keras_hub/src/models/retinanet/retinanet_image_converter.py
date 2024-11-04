import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.RetinaNetImageConverter")
class RetinaNetImageConverter(ImageConverter):
    """Preprocesses images for RetinaNet.

    This layer performs image scaling, offsetting, and normalization using the
    ImageNet mean and standard deviation. It prepares images for use with the
    RetinaNet model.

    Args:
        scale: float. Scaling factor to apply to the image.
            Defaults to `None` (no scaling).
        offset: float. Offset value to add to the image.
            Defaults to `None` (no offset).
        norm_mean: List of 3 floats representing the mean
            values for RGB channels for normalization. Defaults to ImageNet mean
            `[0.485, 0.456, 0.406]`.
        norm_std: List of 3 floats representing the standard
            deviation values for RGB channels for normalization. Defaults to
            ImageNet std `[0.229, 0.224, 0.225]`.
        data_format: str. One of `channels_last` (default) or
            `channels_first`. The ordering of the dimensions in the inputs
    """

    backbone_cls = RetinaNetBackbone

    def __init__(
        self,
        scale=None,
        offset=None,
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
        self.rescale = keras.layers.Rescaling(
            scale=1.0 if scale is None else scale,
            offset=0.0 if offset is None else offset,
        )
        self.data_format = standardize_data_format(data_format)
        self.normalize = keras.layers.Normalization(
            axis=1 if self.data_format == "channels_first" else -1,
            mean=norm_mean,
            variance=(
                [val**2 for val in norm_std] if norm_std is not None else None
            ),
        )
        # TODO: Add resizing height and widh in place of (None, None, 3)
        self.normalize.build((None, None, 3))
        self.built = True

    @preprocessing_function
    def call(self, inputs):
        # TODO: Add resizing https://github.com/keras-team/keras-hub/issues/1965
        x = inputs
        # Rescaling Image
        x = self.rescale(x)
        # By default normalize using imagenet mean and std
        x = self.normalize(x)
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
