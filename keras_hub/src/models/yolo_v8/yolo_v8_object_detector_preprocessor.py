from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_object_detector_preprocessor import (
    ImageObjectDetectorPreprocessor,
)
from keras_hub.src.models.yolo_v8.yolo_v8_backbone import YOLOV8Backbone
from keras_hub.src.models.yolo_v8.yolo_v8_image_converter import (
    YOLOV8ImageConverter,
)


@keras_hub_export("keras_hub.models.YOLOV8ObjectDetectorPreprocessor")
class YOLOV8ObjectDetectorPreprocessor(ImageObjectDetectorPreprocessor):
    backbone_cls = YOLOV8Backbone
    image_converter_cls = YOLOV8ImageConverter
