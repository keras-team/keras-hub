import math
import warnings

import numpy as np

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
    """Image preprocessor for Qwen2-VL.

    Converts a raw NumPy image (H, W, C) or a list of frames into the flat
    patch tensor and ``grid_thw`` metadata required by
    ``Qwen2VLVisionEncoder``.

    Processing steps:
    1. Smart-resize to dimensions divisible by ``patch_size * merge_size``.
    2. Rescale pixel values to ``[0, 1]``.
    3. Normalize with CLIP mean/std.
    4. Pad temporal dimension to a multiple of ``temporal_patch_size``.
    5. Reshape into flat patches of shape
       ``(grid_t * grid_h * grid_w,
         in_channels * temporal_patch_size * patch_size²)``.

    Returns a tuple ``(patches, grid_thw)`` where ``grid_thw`` is a
    NumPy array of shape ``(num_images, 3)`` with ``[grid_t, grid_h, grid_w]``.

    Args:
        min_pixels: int. Minimum total pixel count after resize.
            Defaults to ``56 * 56``.
        max_pixels: int. Maximum total pixel count after resize.
            Defaults to ``12845056``.
        patch_size: int. Spatial patch size. Defaults to ``14``.
        temporal_patch_size: int. Temporal patch size. Defaults to ``2``.
        merge_size: int. Spatial merge factor (used to compute the resize
            factor ``patch_size * merge_size``). Defaults to ``2``.
        image_mean: list of float. Per-channel normalization mean.
            Defaults to CLIP mean.
        image_std: list of float. Per-channel normalization std.
            Defaults to CLIP std.

    Note:
        The ``dtype`` is always forced to ``float32`` for preprocessing
        regardless of any value passed by the caller. A warning is emitted
        if a non-default ``dtype`` is supplied.
    """

    backbone_cls = Qwen2VLBackbone

    def __init__(
        self,
        min_pixels=56 * 56,
        max_pixels=12845056,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        **kwargs,
    ):
        # Force float32 for image preprocessing.
        user_dtype = kwargs.pop("dtype", None)
        if user_dtype is not None:
            warnings.warn(
                f"Qwen2VLImageConverter forces dtype='float32' for "
                f"preprocessing. The supplied dtype='{user_dtype}' "
                f"will be ignored.",
                stacklevel=2,
            )
        super().__init__(dtype="float32", **kwargs)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = np.array(image_mean, dtype="float32")
        self.image_std = np.array(image_std, dtype="float32")
        self._factor = patch_size * merge_size

    def call(self, image):
        """Preprocess a single image, a list of video frames, or
        a list of separate images.

        Args:
            image: NumPy array of shape ``(H, W, C)`` for a single image,
                ``(T, H, W, C)`` for a video clip, or a **list** of
                NumPy arrays (each ``(H, W, C)``) for multiple separate
                images. Pixel values should be in ``[0, 255]``.

        Returns:
            Tuple ``(patches, grid_thw)``:
            - ``patches``: float32 NumPy array of shape
              ``(total_patches,
                C * temporal_patch_size * patch_size²)``.
            - ``grid_thw``: int32 NumPy array of shape
              ``(num_images, 3)`` with ``[grid_t, grid_h, grid_w]``
              per image.
        """
        # Handle a list of separate images by processing each one
        # independently and concatenating the results.
        if isinstance(image, list):
            all_patches = []
            all_grids = []
            for img in image:
                p, g = self.call(img)
                all_patches.append(p)
                all_grids.append(g)
            return (
                np.concatenate(all_patches, axis=0),
                np.concatenate(all_grids, axis=0),
            )

        image = np.asarray(image, dtype="float32")
        if image.ndim == 3:
            image = image[np.newaxis]

        height, width = image.shape[1], image.shape[2]
        resized_h, resized_w = smart_resize(
            height,
            width,
            factor=self._factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        frames = []
        for frame in image:
            if resized_h != height or resized_w != width:
                frame = self._resize_frame(frame, resized_h, resized_w)
            frames.append(frame)

        patches = np.stack(frames, axis=0)

        patches = patches / np.float32(255.0)
        patches = (patches - self.image_mean) / self.image_std

        patches = patches.transpose(0, 3, 1, 2)

        num_frames = patches.shape[0]
        if num_frames % self.temporal_patch_size != 0:
            repeat = self.temporal_patch_size - (
                num_frames % self.temporal_patch_size
            )
            patches = np.concatenate(
                [patches, np.repeat(patches[-1:], repeat, axis=0)], axis=0
            )

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size,
        )

        grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype="int32")
        return patches, grid_thw

    def _resize_frame(self, frame, target_h, target_w):
        """Resize a single frame using PIL (preferred) or NumPy fallback."""
        try:
            from PIL import Image as PILImage

            if hasattr(PILImage, "Resampling"):
                resample = PILImage.Resampling.BICUBIC
            else:
                resample = PILImage.BICUBIC
            pil = PILImage.fromarray(frame.astype("uint8"))
            pil = pil.resize((target_w, target_h), resample)
            return np.array(pil, dtype="float32")
        except ImportError:
            return _numpy_resize(frame, target_h, target_w)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "merge_size": self.merge_size,
                "image_mean": list(self.image_mean),
                "image_std": list(self.image_std),
            }
        )
        return config


def _numpy_resize(frame, new_h, new_w):
    """Fallback bilinear resize using NumPy (no PIL dependency)."""
    old_h, old_w = frame.shape[:2]
    row_idx = np.linspace(0, old_h - 1, new_h)
    col_idx = np.linspace(0, old_w - 1, new_w)
    r0 = np.floor(row_idx).astype(int).clip(0, old_h - 1)
    r1 = np.ceil(row_idx).astype(int).clip(0, old_h - 1)
    c0 = np.floor(col_idx).astype(int).clip(0, old_w - 1)
    c1 = np.ceil(col_idx).astype(int).clip(0, old_w - 1)
    dr = (row_idx - r0)[:, np.newaxis, np.newaxis]
    dc = (col_idx - c0)[np.newaxis, :, np.newaxis]
    top = frame[r0][:, c0] * (1 - dc) + frame[r0][:, c1] * dc
    bot = frame[r1][:, c0] * (1 - dc) + frame[r1][:, c1] * dc
    return (top * (1 - dr) + bot * dr).astype("float32")
