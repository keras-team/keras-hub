from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.depth_anything.depth_anything_backbone import (
    DepthAnythingBackbone,
)
from keras_hub.src.models.depth_anything.depth_anything_image_converter import (
    DepthAnythingImageConverter,
)
from keras_hub.src.models.depth_estimator_preprocessor import (
    DepthEstimatorPreprocessor,
)


@keras_hub_export("keras_hub.models.DepthAnythingDepthEstimatorPreprocessor")
class DepthAnythingDepthEstimatorPreprocessor(DepthEstimatorPreprocessor):
    backbone_cls = DepthAnythingBackbone
    image_converter_cls = DepthAnythingImageConverter
