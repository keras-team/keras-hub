from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.siglip.siglip_backbone import SigLIPBackbone


@keras_hub_export("keras_hub.layers.SigLIPImageConverter")
class SigLIPImageConverter(ImageConverter):
    backbone_cls = SigLIPBackbone
