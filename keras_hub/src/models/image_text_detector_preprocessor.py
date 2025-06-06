import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.diffbin.db_utils import get_mask
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
        """Converts polygon/bounding box labels to a binary mask.
        Pixel within text region is 1, otherwise 0.
        Args:
            x: Input image.
            y: Input label (polygon or bounding box).
            sample_weight: Sample weight for the input data.
        Returns:
            A tuple of preprocessed image and its corresponding binary mask.
        """
        if y is None:
            return self.image_converter(x)
        else:
            x = self.image_converter(x)
            target_h, target_w = self.image_size

            original_w, original_h = self.annotation_size

            scale_x = target_w / original_w
            scale_y = target_h / original_h
            polys = y["polygons"]
            ignores = y["ignores"]

            scaled_polygons = [
                [
                    (float(pt[0]) * scale_x, float(pt[1]) * scale_y)
                    for pt in poly
                ]
                for poly in polys
            ]
            mask = get_mask(target_w, target_h, scaled_polygons, ignores)

        return keras.utils.pack_x_y_sample_weight(x, mask, sample_weight)
