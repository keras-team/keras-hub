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
    CLIP-mean/std normalization, temporal padding, patch reshape.

    Args:
        min_pixels: int. Minimum total pixel count after resize.
            Defaults to ``3136``.
        max_pixels: int. Maximum total pixel count after resize.
            Defaults to ``12845056``.
        patch_size: int. Spatial patch size. Defaults to ``14``.
        temporal_patch_size: int. Temporal patch size. Defaults to ``2``.
        merge_size: int. Spatial merge factor. Defaults to ``2``.
        image_mean: list of float. Per-channel normalization mean.
            Defaults to the CLIP mean ``(0.48145466, 0.4578275,
            0.40821073)``.
        image_std: list of float. Per-channel normalization std.
            Defaults to the CLIP std ``(0.26862954, 0.26130258,
            0.27577711)``.

    Examples:

    ```python
    # Unbatched input.
    converter = keras_hub.layers.Qwen2_5VLImageConverter()
    converter(np.random.randint(0, 256, size=(224, 224, 3)))

    # Batched input (multiple images).
    converter(
        [np.random.randint(0, 256, size=(224, 224, 3)),
         np.random.randint(0, 256, size=(448, 224, 3))]
    )
    ```
    """

    backbone_cls = Qwen2_5VLBackbone
