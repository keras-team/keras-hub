import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_segmenter_preprocessor import (
    ImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_image_converter import SAMImageConverter
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.SAMImageSegmenterPreprocessor")
class SAMImageSegmenterPreprocessor(ImageSegmenterPreprocessor):
    backbone_cls = SAMBackbone
    image_converter_cls = SAMImageConverter

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        images = x["images"]
        if self.image_converter:
            x["images"] = self.image_converter(images)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
