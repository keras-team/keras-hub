import cv2
import keras
import numpy as np

from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


class ImageTextDetectorPreprocessor(Preprocessor):
    """Base class for image text detector preprocessing layers."""

    def __init__(
        self,
        image_converter=None,
        target_size=(640, 640),
        shrink_ratio=0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter
        self.target_size = target_size
        self.shrink_ratio = shrink_ratio

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        if y is None:
            return self.image_converter(x)
        else:
            # Pass bounding boxes through image converter in the dictionary
            # with keys format standardized by core Keras.
            output = self.image_converter(
                {
                    "images": x,
                    "bounding_boxes": y,
                }
            )
            x = output["images"]
            y = output["bounding_boxes"]

            # Text Region= Converts polygon/bounding box labels to a binary mask.
            # Pixel within text region is 1, otherwise 0
            x = self.image_converter(x)

            # Intialize empty mask with zeros
            mask = np.zeros(x.shape[:2], dtype=np.uint8)
            for poly in y:
                poly = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [poly], 1)

            # check if edge pixels are 1
            top_edge = np.any(mask[0, :])
            bottom_edge = np.any(mask[-1, :])
            left_edge = np.any(mask[:, 0])
            right_edge = np.any(mask[:, -1])

            if not (top_edge or bottom_edge or left_edge or right_edge):
                # Shrink the mask by a ratio
                y = cv2.resize(
                    y,
                    (
                        int(mask.shape[1] * self.shrink_ratio),
                        int(mask.shape[0] * self.shrink_ratio),
                    ),
                )
            else:
                y = mask

            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
