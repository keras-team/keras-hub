import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.ImageTextDetectorPreprocessor")
class ImageTextDetectorPreprocessor(Preprocessor):
    """Base class for image text detector preprocessing layers.
    This class is used to preprocess images and their corresponding labels
    for training and inference. It converts polygon/bounding box labels to a
    binary mask, where pixels within the text region are set to 1 and
    pixels outside the text region are set to 0.
    Args:
        image_converter: A callable that converts images to the desired format.
        image_size: A tuple specifying the target size of the images.
        **kwargs: Additional keyword arguments.

    Examples:
        ```python
        preprocessor = ImageTextDetectorPreprocessor(
            image_converter=my_image_converter,
            image_size=(640, 640)
        )
        ```
    Returns:
        A preprocessed image and its corresponding binary mask.
    """

    def __init__(
        self,
        image_converter=None,
        image_size=(640, 640),
        annotation_size=(640, 360),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter
        self.image_size = image_size
        self.annotation_size = annotation_size

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        """Preprocess the input image and its corresponding label.
        Args:
            x: Input image tensor.
            y: Optional label tensor, can be a polygon or bounding box.
            sample_weight: Optional sample weight tensor.
        Returns:
            A tuple of preprocessed image and binary mask if `y` is provided,
            otherwise just the preprocessed image.
        """
        x = self.image_converter(x)
        if y is None:
            return self.image_converter(x)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
