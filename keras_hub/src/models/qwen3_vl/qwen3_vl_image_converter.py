import math

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs_grain
from keras_hub.src.utils.tensor_utils import in_grain_data_pipeline
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

# torch's integer resize keeps weights in at most 22 fractional bits, like
# PIL, so that a sum of `uint8` pixels times weights fits in an int32.
_MAX_PRECISION = 22


def _cubic(x):
    """The `a=-0.5` cubic kernel used by PIL and torch's antialiased resize."""
    a = -0.5
    x = np.abs(x)
    near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    far = (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return np.where(x < 1.0, near, np.where(x < 2.0, far, 0.0))


def _bicubic_taps(in_size, out_size):
    """Integer taps of torch's antialiased bicubic resize along one axis.

    Returns:
        A tuple `(indices, weights, precision)`. `indices` and `weights` are
        `(out_size, taps)` int32 arrays. Output `i` is
        `sum(x[indices[i]] * weights[i]) >> precision`, rounded and clipped.
    """
    scale = in_size / out_size
    support = 2.0 * scale if scale >= 1.0 else 2.0
    inv_scale = 1.0 / scale if scale >= 1.0 else 1.0
    center = scale * (np.arange(out_size) + 0.5)
    start = np.maximum(np.trunc(center - support + 0.5), 0).astype("int32")
    stop = np.minimum(np.trunc(center + support + 0.5), in_size)
    stop = stop.astype("int32")
    taps = int((stop - start).max())
    indices = start[:, None] + np.arange(taps, dtype="int32")
    weights = _cubic((indices - center[:, None] + 0.5) * inv_scale)
    weights = np.where(indices < stop[:, None], weights, 0.0)
    # Sum the taps in order, as torch does, so the normalized weights round
    # to the same integers.
    total = np.zeros(out_size)
    for k in range(taps):
        total = total + weights[:, k]
    weights = weights / np.where(total == 0.0, 1.0, total)[:, None]
    max_weight = weights.max()
    precision = 0
    while precision < _MAX_PRECISION:
        if int(0.5 + max_weight * (1 << (precision + 1))) >= (1 << 15):
            break
        precision += 1
    weights = weights * (1 << precision)
    weights = np.trunc(np.where(weights < 0, weights - 0.5, weights + 0.5))
    indices = np.minimum(indices, in_size - 1)
    return indices, weights.astype("int32"), precision


def _resize_axis(x, size, axis):
    """One pass of torch's `uint8` antialiased bicubic resize, in NumPy."""
    if x.shape[axis] == size:
        return x
    indices, weights, precision = _bicubic_taps(x.shape[axis], size)
    x = np.moveaxis(x, axis, -1)
    total = np.full(x.shape[:-1] + (size,), 1 << (precision - 1), "int32")
    for k in range(indices.shape[1]):
        total += x[..., indices[:, k]] * weights[:, k]
    x = np.clip(total >> precision, 0, 255)
    return np.moveaxis(x, -1, axis)


def resize_integer_pixels(x, height, width, interpolation, antialias):
    """Resize integer pixels the way the HF processors do.

    The HF processors hand `uint8` images to torchvision, whose CPU kernel
    resizes one axis at a time, width first, with fixed-point weights and
    rounds back to `uint8` after each pass. For antialiased bicubic resizing
    this function reproduces that kernel exactly, on the host. Other methods
    resize in float and round after each pass.

    Args:
        x: Float batch of images, `(batch, source_height, source_width, 3)`.
        height: Target height.
        width: Target width.
        interpolation: string. Resize method.
        antialias: bool. Whether resizing is antialiased.
    """
    if interpolation == "bicubic" and antialias:
        x = ops.convert_to_numpy(x).astype("int32")
        x = _resize_axis(_resize_axis(x, width, axis=2), height, axis=1)
        return ops.convert_to_tensor(x.astype("float32"))

    def resize(x, size):
        x = ops.image.resize(
            x, size=size, interpolation=interpolation, antialias=antialias
        )
        return ops.round(ops.clip(x, 0.0, 255.0))

    return resize(resize(x, (ops.shape(x)[1], width)), (height, width))


def _cubic_tf(x):
    a = -0.5
    x = tf.abs(x)
    near = ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    far = (((x - 5.0) * x + 8.0) * x - 4.0) * a
    zeros = tf.zeros_like(x)
    return tf.where(x < 1.0, near, tf.where(x < 2.0, far, zeros))


def _resize_axis_tf(x, size, axis):
    """`_resize_axis` on int32 tensors, for sizes known only in the graph."""
    in_size = tf.shape(x)[axis]

    def resize():
        f64 = "float64"
        scale = tf.cast(in_size, f64) / tf.cast(size, f64)
        support = tf.where(scale >= 1.0, 2.0 * scale, 2.0)
        inv_scale = tf.where(scale >= 1.0, 1.0 / scale, 1.0)
        center = scale * (tf.range(size, dtype=f64) + 0.5)
        start = tf.maximum(tf.cast(center - support + 0.5, "int32"), 0)
        stop = tf.minimum(tf.cast(center + support + 0.5, "int32"), in_size)
        taps = tf.reduce_max(stop - start)
        indices = start[:, None] + tf.range(taps)[None, :]
        offsets = tf.cast(indices, f64) - center[:, None] + 0.5
        weights = _cubic_tf(offsets * inv_scale)
        weights = tf.where(
            indices < stop[:, None], weights, tf.zeros_like(weights)
        )
        # `scan` sums the taps in order, like `_bicubic_taps`.
        total = tf.scan(
            lambda acc, w: acc + w,
            tf.transpose(weights),
            initializer=tf.zeros([size], f64),
        )[-1]
        total = tf.where(total == 0.0, tf.ones_like(total), total)
        weights = weights / total[:, None]
        # The largest precision below the cap whose weights fit in int16.
        steps = tf.range(_MAX_PRECISION, dtype=f64)
        fits = tf.floor(0.5 + tf.reduce_max(weights) * 2.0 ** (steps + 1))
        precision = tf.reduce_sum(tf.cast(fits < 2.0**15, "int32"))
        weights = weights * tf.cast(tf.bitwise.left_shift(1, precision), f64)
        weights = tf.where(weights < 0, weights - 0.5, weights + 0.5)
        weights = tf.cast(weights, "int32")
        indices = tf.minimum(indices, in_size - 1)

        perm = list(range(4))
        perm[axis], perm[3] = 3, axis
        moved = tf.transpose(x, perm)
        init = tf.bitwise.left_shift(1, precision - 1) * tf.ones(
            tf.concat([tf.shape(moved)[:-1], [size]], 0), "int32"
        )

        def step(k, total):
            pixels = tf.gather(moved, indices[:, k], axis=3)
            return k + 1, total + pixels * weights[:, k]

        _, total = tf.while_loop(
            lambda k, _: k < taps, step, [tf.constant(0), init]
        )
        out = tf.clip_by_value(tf.bitwise.right_shift(total, precision), 0, 255)
        return tf.transpose(out, perm)

    return tf.cond(tf.equal(in_size, size), lambda: x, resize)


def resize_integer_pixels_tf(x, height, width, method, antialias):
    """`resize_integer_pixels` for `tf.data` graphs."""
    if method == "bicubic" and antialias:
        x = tf.cast(x, "int32")
        x = _resize_axis_tf(_resize_axis_tf(x, width, 2), height, 1)
        return tf.cast(x, "float32")

    def resize(x, size):
        x = tf.image.resize(x, size, method=method, antialias=antialias)
        return tf.round(tf.clip_by_value(x, 0.0, 255.0))

    return resize(resize(x, (tf.shape(x)[1], width)), (height, width))


def eager_tf_preprocessing():
    """Whether eager preprocessing runs through TensorFlow.

    On JAX every new image or video shape compiles the resize again, which
    costs hundreds of milliseconds per shape. TensorFlow runs eagerly without
    compiling.
    """
    return tf is not None and keras.config.backend() == "jax"


def smart_resize_volume(
    num_frames,
    height,
    width,
    temporal_factor,
    factor,
    min_pixels,
    max_pixels,
    check_min_size=True,
):
    """Python port of the HF Qwen3-VL `smart_resize`.

    The pixel budget is checked against `frames * height * width`, with the
    frame count rounded up to a multiple of `temporal_factor`.

    A single image is `num_frames=1, temporal_factor=1`. HF only rejects
    sides smaller than `factor` for videos; images that small are upscaled,
    so `smart_resize` passes `check_min_size=False`.

    Returns:
        A `(height, width)` tuple of ints.
    """
    if min(num_frames, height, width) <= 0:
        raise ValueError(
            "`num_frames`, `height` and `width` must be positive. Received: "
            f"num_frames={num_frames}, height={height}, width={width}"
        )
    if check_min_size and (height < factor or width < factor):
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
    # `max_pixels` may be a tensor when it depends on the frame count.
    max_pixels = tf.cast(max_pixels, f64)

    def shrink():
        beta = tf.sqrt(n * h * w / max_pixels)
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
        volume > max_pixels,
        shrink,
        lambda: tf.cond(
            volume < float(min_pixels), grow, lambda: (h_bar, w_bar)
        ),
    )
    return tf.cast(h_bar, "int32"), tf.cast(w_bar, "int32")


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Python port of the HF `smart_resize` for single images."""
    return smart_resize_volume(
        1,
        height,
        width,
        1,
        factor,
        min_pixels,
        max_pixels,
        check_min_size=False,
    )


def _smart_resize_tf(height, width, factor, min_pixels, max_pixels):
    """Graph-mode version of `smart_resize` on scalar int tensors."""
    return smart_resize_volume_tf(
        1, height, width, 1, factor, min_pixels, max_pixels
    )


@keras_hub_export("keras_hub.layers.Qwen3VLImageConverter")
class Qwen3VLImageConverter(ImageConverter):
    """Image converter for Qwen3-VL.

    Resizes an image with the Qwen `smart_resize` rule, normalizes it, and
    cuts it into the flattened patch tensor consumed by
    `Qwen3VLVisionEncoder`. Patches are emitted in merge-block order: for
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
        repeat_frames: bool. Whether to repeat the image
            `temporal_patch_size` times along the time axis, as HF does.
            The vision encoder also accepts a single frame and repeats it on
            the device. Defaults to `True`.

    Returns:
        A dict with `"patches"`, a float tensor of shape
        `(num_patches, temporal_patch_size, patch_size, patch_size, 3)`, or
        `(num_patches, 1, patch_size, patch_size, 3)` without
        `repeat_frames`, and
        `"grid_thw"`, the int32 vector `[1, height // patch_size,
        width // patch_size]`.
    """

    backbone_cls = Qwen3VLBackbone

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
    def call(self, inputs, repeat_frames=True):
        if in_tf_function():
            return self._call_tf(inputs, repeat_frames)
        if eager_tf_preprocessing():
            outputs = self._call_tf(tf.convert_to_tensor(inputs), repeat_frames)
        else:
            outputs = self._call_ops(inputs, repeat_frames)
        if in_grain_data_pipeline():
            # Grain pickles outputs across worker processes, so return NumPy
            # arrays rather than backend tensors.
            return convert_preprocessing_outputs_grain(outputs)
        return outputs

    def _grid_thw(self, height, width):
        """The `grid_thw` of an image, computed on the host."""
        target_h, target_w = smart_resize(
            height, width, self._factor, self.min_pixels, self.max_pixels
        )
        return [1, target_h // self.patch_size, target_w // self.patch_size]

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

    def _call_tf(self, inputs, repeat_frames=True):
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
        image = tf.clip_by_value(image, 0.0, 255.0)
        # Normalize with TF constants. `_normalize` uses backend ops, which
        # on JAX would copy the image to the device and back.
        if self.scale is not None:
            image = image * tf.reshape(tf.constant(self.scale, "float32"), [-1])
        if self.offset is not None:
            image = image + tf.reshape(
                tf.constant(self.offset, "float32"), [-1]
            )
        image = tf.cast(image, self.compute_dtype)
        p, m = self.patch_size, self.spatial_merge_size
        grid_h = target_h // p
        grid_w = target_w // p
        image = tf.reshape(image, (grid_h // m, m, p, grid_w // m, m, p, 3))
        image = tf.transpose(image, (0, 3, 1, 4, 2, 5, 6))
        image = tf.reshape(image, (grid_h * grid_w, 1, p, p, 3))
        if repeat_frames:
            image = tf.tile(image, [1, self.temporal_patch_size, 1, 1, 1])
        grid_thw = tf.stack([tf.constant(1, "int32"), grid_h, grid_w])
        return {"patches": image, "grid_thw": grid_thw}

    def _call_ops(self, inputs, repeat_frames=True):
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
        if repeat_frames:
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
