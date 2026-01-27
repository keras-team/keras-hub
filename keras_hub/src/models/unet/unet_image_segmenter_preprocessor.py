import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)


@keras_hub_export("keras_hub.models.UNetImageSegmenterPreprocessor")
class UNetImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    """Preprocessor for UNet image segmentation.

    This preprocessor passes through the input images and labels without
    modification. UNet accepts variable input sizes and handles raw image
    data directly. No specific preprocessing (normalization, resizing, etc.)
    is applied by default.

    If you need custom preprocessing, you can subclass this and override
    the `call` method, or provide a custom `image_converter`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Default preprocessor (pass-through)
    preprocessor = keras_hub.models.UNetImageSegmenterPreprocessor()

    # Usage with images
    images = np.random.uniform(0, 1, size=(2, 256, 256, 3))
    labels = np.random.randint(0, 2, size=(2, 256, 256, 1))
    preprocessed = preprocessor(images, labels)
    ```
    """

    backbone_cls = None  # Set in __init__ to avoid circular import

    def __init__(self, **kwargs):
        # UNet doesn't require a specific image converter
        super().__init__(image_converter=None, **kwargs)

    def call(self, x, y=None, sample_weight=None):
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
