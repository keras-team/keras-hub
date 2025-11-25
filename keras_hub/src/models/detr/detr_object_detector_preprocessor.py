from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.detr.detr_backbone import DETRBackbone
from keras_hub.src.models.detr.detr_image_converter import DETRImageConverter
from keras_hub.src.models.object_detector_preprocessor import (
    ObjectDetectorPreprocessor,
)


@keras_hub_export("keras_hub.models.DETRObjectDetectorPreprocessor")
class DETRObjectDetectorPreprocessor(ObjectDetectorPreprocessor):
    backbone_cls = DETRBackbone
    image_converter_cls = DETRImageConverter
