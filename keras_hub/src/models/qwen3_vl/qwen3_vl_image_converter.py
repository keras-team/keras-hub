import math

from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs_grain
from keras_hub.src.utils.tensor_utils import in_grain_data_pipeline
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


def resize_integer_pixels(x, height, width, interpolation, antialias):
    """Resize integer pixels the way PIL and torchvision do.

    The HF processors resize `uint8` images one axis at a time, width first,
    and round back to integers after each pass. A single float resize leaves
    fractional values and differs from HF by up to one pixel level, which is
    enough to change the logits of a vision-language model.

    Args:
        x: Float batch of images, `(batch, source_height, source_width, 3)`.
        height: Target height.
        width: Target width.
        interpolation: string. Resize method.
        antialias: bool. Whether resizing is antialiased.
    """

    def resize(x, size):
        x = ops.image.resize(
            x, size=size, interpolation=interpolation, antialias=antialias
        )
        return ops.round(ops.clip(x, 0.0, 255.0))

    return resize(resize(x, (ops.shape(x)[1], width)), (height, width))


def resize_integer_pixels_tf(x, height, width, method, antialias):
    """`resize_integer_pixels` for `tf.data` graphs."""

    def resize(x, size):
        x = tf.image.resize(x, size, method=method, antialias=antialias)
        return tf.round(tf.clip_by_value(x, 0.0, 255.0))

    return resize(resize(x, (tf.shape(x)[1], width)), (height, width))


def smart_resize_volume(
    num_frames, height, width, temporal_factor, factor, min_pixels, max_pixels
):
    """Python port of the HF Qwen3-VL `smart_resize`.

    The pixel budget is checked against `frames * height * width`, with the
    frame count rounded up to a multiple of `temporal_factor`.

    A single image is `num_frames=1, temporal_factor=1`.

    Returns:
        A `(height, width)` tuple of ints.
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be at least "
            f"factor:{factor}."
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "Absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}."
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_resize_volume_tf(
    num_frames, height, width, temporal_factor, factor, min_pixels, max_pixels
):
    """Graph-mode version of `smart_resize_volume` on scalar int tensors."""
    f64 = "float64"
    n = tf.cast(num_frames, f64)
    h = tf.cast(height, f64)
    w = tf.cast(width, f64)
    fac = tf.constant(float(factor), f64)
    h_bar = tf.round(h / fac) * fac
    w_bar = tf.round(w / fac) * fac
    t_bar = tf.math.ceil(n / temporal_factor) * temporal_factor
    volume = t_bar * h_bar * w_bar

    def shrink():
        beta = tf.sqrt(n * h * w / float(max_pixels))
        return (
            tf.maximum(fac, tf.floor(h / beta / fac) * fac),
            tf.maximum(fac, tf.floor(w / beta / fac) * fac),
        )

    def grow():
        beta = tf.sqrt(float(min_pixels) / (n * h * w))
        return (
            tf.math.ceil(h * beta / fac) * fac,
            tf.math.ceil(w * beta / fac) * fac,
        )

    h_bar, w_bar = tf.cond(
        volume > float(max_pixels),
        shrink,
        lambda: tf.cond(
            volume < float(min_pixels), grow, lambda: (h_bar, w_bar)
        ),
    )
    return tf.cast(h_bar, "int32"), tf.cast(w_bar, "int32")


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Python port of the HF `smart_resize` for single images."""
    return smart_resize_volume(
        1, height, width, 1, factor, min_pixels, max_pixels
    )


def _smart_resize_tf(height, width, factor, min_pixels, max_pixels):
    """Graph-mode version of `smart_resize` on scalar int tensors."""
    return smart_resize_volume_tf(
        1, height, width, 1, factor, min_pixels, max_pixels
    )


@keras_hub_export("keras_hub.layers.Qwen3_VLImageConverter")
class Qwen3_VLImageConverter(ImageConverter):
    """Image converter for Qwen3-VL.

    Resizes an image with the Qwen `smart_resize` rule, normalizes it, and
    cuts it into the flattened patch tensor consumed by
    `Qwen3_VLVisionEncoder`. Patches are emitted in merge-block order: for
    each `spatial_merge_size x spatial_merge_size` block of patches, the
    patches are consecutive, and blocks are ordered row by row.

    Args:
        patch_size: int. Side of a square patch in pixels. Defaults to `16`.
        temporal_patch_size: int. Number of frames per patch. A still image is
            repeated along this axis. Defaults to `2`.
        spatial_merge_size: int. Side of the patch block merged into one
            token. Defaults to `2`.
        min_pixels: int. Images with fewer pixels than this are upscaled.
            Defaults to `65536` (`shortest_edge` in the HF config).
        max_pixels: int. Images with more pixels than this are downscaled.
            Defaults to `16777216` (`longest_edge` in the HF config).
        scale: float or list of floats. Multiplier applied to pixels in
            `[0, 255]`. Defaults to `2 / 255`.
        offset: float or list of floats. Value added after scaling. The
            defaults map pixels to `[-1, 1]`, which is the HF
            `mean=std=0.5` normalization. Defaults to `-1.0`.
        interpolation: string. Resize method. Defaults to `"bicubic"`.
        antialias: bool. Whether resizing is antialiased. Defaults to `True`.
        crop_to_aspect_ratio: bool. Defaults to `False`, since the target
            size already preserves the aspect ratio.

    Call arguments:
        inputs: A `(height, width, 3)` image with values in `[0, 255]`.

    Returns:
        A dict with `"patches"`, a float tensor of shape
        `(num_patches, temporal_patch_size, patch_size, patch_size, 3)`, and
        `"grid_thw"`, the int32 vector `[1, height // patch_size,
        width // patch_size]`.
    """

    backbone_cls = Qwen3_VLBackbone

    def __init__(
        self,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        scale=2.0 / 255.0,
        offset=-1.0,
        interpolation="bicubic",
        antialias=True,
        crop_to_aspect_ratio=False,
        **kwargs,
    ):
        super().__init__(
            scale=scale,
            offset=offset,
            interpolation=interpolation,
            antialias=antialias,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            **kwargs,
        )
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._factor = patch_size * spatial_merge_size

    @preprocessing_function
    def call(self, inputs):
        if in_tf_function():
            return self._call_tf(inputs)
        outputs = self._call_ops(inputs)
        if in_grain_data_pipeline():
            # Grain pickles outputs across worker processes, so return NumPy
            # arrays rather than backend tensors.
            return convert_preprocessing_outputs_grain(outputs)
        return outputs

    def _normalize(self, image):
        if self.scale is not None:
            scale = self._expand_non_channel_dims(self.scale, image)
            image, scale = self._convert_types(image, scale, self.compute_dtype)
            image = image * scale
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, image)
            image, offset = self._convert_types(image, offset, image.dtype)
            image = image + offset
        return image

    def _call_tf(self, inputs):
        image = tf.cast(inputs, "float32")
        target_h, target_w = _smart_resize_tf(
            tf.shape(image)[0],
            tf.shape(image)[1],
            self._factor,
            self.min_pixels,
            self.max_pixels,
        )
        if inputs.dtype.is_integer:
            image = resize_integer_pixels_tf(
                image[tf.newaxis],
                target_h,
                target_w,
                self.interpolation,
                self.antialias,
            )[0]
        else:
            image = tf.image.resize(
                image[tf.newaxis],
                (target_h, target_w),
                method=self.interpolation,
                antialias=self.antialias,
            )[0]
        image = self._normalize(tf.clip_by_value(image, 0.0, 255.0))
        p, m = self.patch_size, self.spatial_merge_size
        grid_h = target_h // p
        grid_w = target_w // p
        image = tf.reshape(image, (grid_h // m, m, p, grid_w // m, m, p, 3))
        image = tf.transpose(image, (0, 3, 1, 4, 2, 5, 6))
        image = tf.reshape(image, (grid_h * grid_w, 1, p, p, 3))
        image = tf.tile(image, [1, self.temporal_patch_size, 1, 1, 1])
        grid_thw = tf.stack([tf.constant(1, "int32"), grid_h, grid_w])
        return {"patches": image, "grid_thw": grid_thw}

    def _call_ops(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        is_int = is_int_dtype(inputs.dtype)
        image = ops.cast(inputs, "float32")
        height, width = (int(d) for d in ops.shape(image)[:2])
        target_h, target_w = smart_resize(
            height, width, self._factor, self.min_pixels, self.max_pixels
        )
        image = ops.expand_dims(image, 0)
        if is_int:
            image = resize_integer_pixels(
                image, target_h, target_w, self.interpolation, self.antialias
            )
        else:
            image = ops.image.resize(
                image,
                size=(target_h, target_w),
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
        image = self._normalize(ops.clip(image[0], 0.0, 255.0))
        p, m = self.patch_size, self.spatial_merge_size
        grid_h = target_h // p
        grid_w = target_w // p
        image = ops.reshape(image, (grid_h // m, m, p, grid_w // m, m, p, 3))
        image = ops.transpose(image, (0, 3, 1, 4, 2, 5, 6))
        image = ops.reshape(image, (grid_h * grid_w, 1, p, p, 3))
        image = ops.tile(image, [1, self.temporal_patch_size, 1, 1, 1])
        grid_thw = ops.convert_to_tensor([1, grid_h, grid_w], dtype="int32")
        return {"patches": image, "grid_thw": grid_thw}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }
        )
        return config
