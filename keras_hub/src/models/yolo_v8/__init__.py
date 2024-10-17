from keras_hub.src.models.yolo_v8.yolo_v8_backbone import YOLOV8Backbone
from keras_hub.src.models.yolo_v8.yolo_v8_backbone_presets import (
    backbone_presets,
)
from keras_hub.src.utils.preset_utils import register_presets

register_presets(backbone_presets, YOLOV8Backbone)
