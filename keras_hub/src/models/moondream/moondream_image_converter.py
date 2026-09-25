from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.moondream.moondream_backbone import MoondreamBackbone


@keras_hub_export("keras_hub.layers.MoondreamImageConverter")
class MoondreamImageConverter(ImageConverter):
    """Converts images for the Moondream model.

    This layer resizes and normalizes images to the format expected by the
    Moondream vision encoder. By default, Moondream expects images of size
    `(378, 378)` normalized with the SigLIP normalization constants.

    Args:
        image_size: int or tuple of ints `(height, width)`. The target size
            for the output image. Defaults to `(378, 378)`.
        scale: float or tuple of floats. Scale factor applied to each channel
            of the image after normalization. Defaults to `None` (no scaling
            beyond the `[0, 1]` range rescaling done by `ImageConverter`).
        offset: float or tuple of floats. Offset applied to each channel of
            the image after scaling. Defaults to `None`.

    Examples:
    ```python
    import numpy as np
    import keras_hub

    converter = keras_hub.layers.MoondreamImageConverter()
    images = np.random.rand(2, 512, 512, 3).astype("float32")
    converted = converter(images)
    print(converted.shape)  # (2, 378, 378, 3)
    ```
    """

    backbone_cls = MoondreamBackbone
