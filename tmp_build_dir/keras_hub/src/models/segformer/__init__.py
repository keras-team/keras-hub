from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone
from keras_hub.src.models.segformer.segformer_image_segmenter import (
    SegFormerImageSegmenter,
)
from keras_hub.src.models.segformer.segformer_presets import presets
from keras_hub.src.utils.preset_utils import register_presets

register_presets(presets, SegFormerImageSegmenter)
