
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
            # Convert polygons to binary mask
            mask= get_mask(width,height,poly)
            return keras.utils.pack_x_y_sample_weight(x,mask, sample_weight)
    
    @preprocessing_function
    def generate_postprocess(self,x):
        '''
        Generates postprocess function to convert probability map of 
        model output to polygon
        '''
        probability_maps,threshold_maps = x["probability_maps"], x["threshold_maps"]
        binary_maps = 1.0 / (1.0 + keras.ops.exp(-50.0 * (probability_maps - threshold_maps)))
        outputs = keras.layers.Concatenate(axis=-1)(
            [probability_maps, threshold_maps, binary_maps])
        return outputs
        
    def get_config(self):
        config = super().get_config()
        config.update(
            "target_size": self.target_size,
            "shrink_ratio": self.shrink_ratio
        )
        return config
    

    
            


