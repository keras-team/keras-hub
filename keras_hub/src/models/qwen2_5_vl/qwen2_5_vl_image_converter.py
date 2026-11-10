from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)


@keras_hub_export("keras_hub.layers.Qwen2_5VLImageConverter")
class Qwen2_5VLImageConverter(Qwen2VLImageConverter):
    """Image preprocessor for Qwen2.5-VL.

    Identical pipeline to ``Qwen2VLImageConverter``: smart-resize to
    ``patch_size * merge_size`` multiples, rescale to ``[0, 1]``,
    CLIP-mean/std normalization, temporal padding, patch reshape. HF's
    ``Qwen2_5_VLImageProcessor`` is likewise a direct subclass of
    ``Qwen2VLImageProcessor``.

    Args:
        min_pixels: int. Minimum total pixel count after resize.
        max_pixels: int. Maximum total pixel count after resize.
        patch_size: int. Spatial patch size. Defaults to ``14``.
        temporal_patch_size: int. Temporal patch size. Defaults to ``2``.
        merge_size: int. Spatial merge factor. Defaults to ``2``.
        image_mean: list of float. Per-channel normalization mean.
        image_std: list of float. Per-channel normalization std.
    """

    backbone_cls = Qwen2_5VLBackbone
