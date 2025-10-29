from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.vit_det.vit_det_backbone import ViTDetBackbone


@keras_hub_export("keras_hub.layers.ViTDetImageConverter")
class ViTDetImageConverter(ImageConverter):
    """Image converter for ViTDet models.

    This layer applies ImageNet normalization (mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]) to input images for ViTDet models.

    Args:
        image_size: int or tuple of (height, width). The output size of the
            image. Defaults to `(1024, 1024)`.

    Example:
    ```python
    converter = keras_hub.layers.ViTDetImageConverter(image_size=(1024, 1024))
    converter(np.random.rand(1, 512, 512, 3))  # Resizes and normalizes
    ```
    """

    backbone_cls = ViTDetBackbone

    def __init__(
        self,
        image_size=(1024, 1024),
        **kwargs,
    ):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        variance = [x**2 for x in std]
        super().__init__(
            image_size=image_size,
            scale=1.0 / 255.0,  # Scale to [0, 1]
            mean=mean,
            variance=variance,
            **kwargs,
        )
