from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.detr.detr_backbone import DETRBackbone


@keras_hub_export("keras_hub.layers.DETRImageConverter")
class DETRImageConverter(ImageConverter):
    backbone_cls = DETRBackbone

    def __init__(
        self,
        image_size=(800, 800),
        scale=None,
        offset=None,
        **kwargs,
    ):
        if scale is None:
            # scale = 1/255 / std for each channel
            scale = [
                1.0 / 255.0 / 0.229,  # R channel
                1.0 / 255.0 / 0.224,  # G channel
                1.0 / 255.0 / 0.225,  # B channel
            ]

        if offset is None:
            # offset = -mean / std for each channel
            offset = [
                -0.485 / 0.229,  # R channel
                -0.456 / 0.224,  # G channel
                -0.406 / 0.225,  # B channel
            ]

        super().__init__(
            image_size=image_size,
            scale=scale,
            offset=offset,
            **kwargs,
        )
