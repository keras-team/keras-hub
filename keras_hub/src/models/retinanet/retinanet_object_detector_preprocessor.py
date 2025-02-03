from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.object_detector_preprocessor import (
    ObjectDetectorPreprocessor,
)
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone
from keras_hub.src.models.retinanet.retinanet_image_converter import (
    RetinaNetImageConverter,
)


@keras_hub_export("keras_hub.models.RetinaNetObjectDetectorPreprocessor")
class RetinaNetObjectDetectorPreprocessor(ObjectDetectorPreprocessor):
    backbone_cls = RetinaNetBackbone
    image_converter_cls = RetinaNetImageConverter
