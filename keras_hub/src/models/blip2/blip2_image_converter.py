"""BLIP-2 image converter."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone


@keras_hub_export(
    "keras_hub.layers.BLIP2ImageConverter",
)
class BLIP2ImageConverter(ImageConverter):
    """A preprocessing layer for images used by the BLIP-2 model.

    This converter resizes, normalizes, and rescales images to the format
    expected by the BLIP-2 vision encoder (EVA-CLIP). Preprocessing is always
    performed in float32 regardless of the global Keras dtype policy.

    Preprocessing steps:
    1. **Resizing**: Images are resized to `(224, 224)` (default) using
       bicubic interpolation.
    2. **Normalization**: Pixel values are normalized using EVA-CLIP channel
       statistics (mean and standard deviation per channel).

    Args:
        image_size: `(int, int)` tuple or `None`. The output spatial size of
            the image, not including the channels axis. Defaults to
            `(224, 224)`.
        crop_to_aspect_ratio: bool. If `True`, resize without aspect ratio
            distortion. Defaults to `True`.
        interpolation: string. The interpolation method used during resizing.
            Defaults to `"bicubic"`.
        **kwargs: Standard Keras layer arguments.

    Example:
    ```python
    converter = keras_hub.layers.BLIP2ImageConverter()
    images = np.random.randint(0, 256, (1, 500, 500, 3))
    processed = converter(images)  # shape: (1, 224, 224, 3)
    ```

    References:
        - [Li et al., 2023](https://arxiv.org/abs/2301.12597)
    """

    backbone_cls = BLIP2Backbone

    # EVA-CLIP normalization statistics (mean and std per RGB channel).
    # scale  = 1 / std,  offset = -mean / std
    _SCALE = (
        0.014598426619242919,
        0.015007768493717055,
        0.014220065717024086,
    )
    _OFFSET = (
        -1.79226253374815,
        -1.7520971281645974,
        -1.4802197687835659,
    )

    def __init__(
        self,
        image_size=(224, 224),
        crop_to_aspect_ratio=True,
        interpolation="bicubic",
        **kwargs,
    ):
        # Image preprocessing must always run in float32.
        kwargs.pop("dtype", None)
        super().__init__(
            image_size=image_size,
            scale=self._SCALE,
            offset=self._OFFSET,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            interpolation=interpolation,
            dtype="float32",
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.pop("scale", None)
        config.pop("offset", None)
        return config
