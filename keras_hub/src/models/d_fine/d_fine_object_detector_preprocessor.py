from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone
from keras_hub.src.models.d_fine.d_fine_image_converter import (
    DFineImageConverter,
)
from keras_hub.src.models.object_detector_preprocessor import (
    ObjectDetectorPreprocessor,
)


@keras_hub_export("keras_hub.models.DFineObjectDetectorPreprocessor")
class DFineObjectDetectorPreprocessor(ObjectDetectorPreprocessor):
    backbone_cls = DFineBackbone
    image_converter_cls = DFineImageConverter
