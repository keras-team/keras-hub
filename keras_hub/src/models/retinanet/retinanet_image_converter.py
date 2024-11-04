from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format
from keras_hub.src.utils.tensor_utils import preprocessing_function

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


@keras_hub_export("keras_hub.layers.RetinaNetImageConverter")
class RetinaNetImageConverter(ImageConverter):
    """Preprocesses images for RetinaNet.

    This layer performs image scaling, offsetting, and normalization using the
    ImageNet mean and standard deviation. It prepares images for use with the
    RetinaNet model.

    Args:
        rescale: bool. Rescale image to `[0, 1]` by dividing with `255`.
            Defaults to `True`.
        data_format: str. One of `channels_last` (default) or
            `channels_first`. The ordering of the dimensions in the inputs
    """

    backbone_cls = RetinaNetBackbone

    def __init__(self, rescale=True, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.rescale = rescale
        self.data_format = standardize_data_format(data_format)
        self.built = True

    @preprocessing_function
    def call(self, inputs):
        # TODO: Add resizing https://github.com/keras-team/keras-hub/issues/1965
        x = inputs
        # Rescaling Image
        if self.rescale:
            x = x / 255
        # By default normalize using imagenet mean and std
        x = (x - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rescale": self.rescale,
            }
        )
        return config
