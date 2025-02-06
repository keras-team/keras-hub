from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone


@keras_hub_export("keras_hub.layers.RetinaNetImageConverter")
class RetinaNetImageConverter(ImageConverter):
    backbone_cls = RetinaNetBackbone

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # TODO: update presets and remove these old config options. They were
        # never needed.
        if "norm_mean" in kwargs:
            kwargs["offset"] = [-x for x in kwargs.pop("norm_mean")]
        if "norm_std" in kwargs:
            kwargs["scale"] = [1.0 / x for x in kwargs.pop("norm_std")]
        super().__init__(*args, **kwargs)
