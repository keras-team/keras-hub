import math

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone


def smart_resize(
    height, width, factor=28, min_pixels=56 * 56, max_pixels=12845056
):
    """Resize image dimensions so both are divisible by ``factor`` and the
    total pixel count stays within ``[min_pixels, max_pixels]``.

    Args:
        height: int. Original image height.
        width: int. Original image width.
        factor: int. Both output dims must be multiples of this value.
            Defaults to ``28`` (``patch_size * merge_size = 14 * 2``).
        min_pixels: int. Minimum total pixel count. Defaults to
            ``56 * 56 = 3136``.
        max_pixels: int. Maximum total pixel count. Defaults to
            ``12845056`` (matching HuggingFace).

    Returns:
        Tuple ``(h_bar, w_bar)`` of resized dimensions.

    Raises:
        ValueError: If the absolute aspect ratio exceeds 200.
    """
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Height and width must be positive, "
            f"got height={height}, width={width}."
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width):.1f}."
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@keras_hub_export("keras_hub.layers.Qwen2VLImageConverter")
class Qwen2VLImageConverter(ImageConverter):
    """Image converter for Qwen2-VL models.

    This layer handles image preprocessing (resize, normalize) for the
    Qwen2-VL vision encoder. Image processing is always performed in
    ``float32``.

    The ``smart_resize`` utility (defined above) can be used to compute
    target dimensions that are divisible by the patch/merge factor before
    passing images to this converter.

    Args:
        **kwargs: Keyword arguments passed to the base ``ImageConverter``,
            including ``height``, ``width``, ``scale``, ``offset``,
            ``crop_to_aspect_ratio``, ``interpolation``, etc.
    """

    backbone_cls = Qwen2VLBackbone

    def __init__(self, **kwargs):
        # Always do image preprocessing in float32.
        kwargs.pop("dtype", None)
        dtype = "float32"
        super().__init__(dtype=dtype, **kwargs)
