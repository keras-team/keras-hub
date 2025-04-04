from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.yolo_v8.yolo_v8_backbone import YOLOV8Backbone


@keras_hub_export("keras_hub.layers.YOLOV8ImageConverter")
class YOLOV8ImageConverter(ImageConverter):
    backbone_cls = YOLOV8Backbone
