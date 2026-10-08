from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.d_fine.d_fine_backbone import DFineBackbone


@keras_hub_export("keras_hub.layers.DFineImageConverter")
class DFineImageConverter(ImageConverter):
    backbone_cls = DFineBackbone
