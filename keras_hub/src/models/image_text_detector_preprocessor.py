
import keras
import numpy as np

from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.models.diffbin.db_utils import get_region_coordinate,get_mask


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
        '''
        Converts polygon/bounding box labels to a binary mask. 
        Pixel within text region is 1, otherwise 0
        '''
        if y is None:
            return self.image_converter(x)
        else:
            
            x = self.image_converter(x)

            #get polygans annotations
            width,height= self.target_size
            poly= y["polygons"]
            region_coordinates= get_region_coordinate(x, width,height, 
                                                        self.shrink_ratio)
            # Convert polygons to binary mask
            mask= get_mask(width,height,region_coordinates)
            return keras.utils.pack_x_y_sample_weight(x,mask, sample_weight)


            


