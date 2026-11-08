import math

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs_grain
from keras_hub.src.utils.tensor_utils import in_grain_data_pipeline
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


def _resize_output_size_rescale_to_max_len(height, width, max_len):
    """Resize so the longest edge = max_len, preserving aspect ratio."""
    aspect_ratio = width / height
    if width >= height:
        width = max_len
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    else:
        height = max_len
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1
    height = max(height, 1)
    width = max(width, 1)
    return height, width


def _resize_output_size_scale_below_upper_bound(height, width, max_len=4096):
    """Scale down if either dimension exceeds max_len."""
    aspect_ratio = width / height
    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, 1)
    width = max(width, 1)
    return height, width


def _resize_tf(images, size, method="lanczos3", antialias=True):
    images = tf.image.resize(
        images,
        size=(tf.shape(images)[1], size[1]),
        method=method,
        antialias=antialias,
    )
    images = tf.round(tf.clip_by_value(images, 0.0, 255.0))
    images = tf.image.resize(
        images,
        size=(size[0], size[1]),
        method=method,
        antialias=antialias,
    )
    return tf.round(tf.clip_by_value(images, 0.0, 255.0))


# Pillow's `Resample.c` works in fixed point for 8-bit images: the filter
# taps are normalized in double precision, quantized to `PRECISION_BITS`
# (32 - 8 - 2 = 22), accumulated in int32 from a half-unit offset, shifted
# back down and clipped to uint8 after *each* pass (width, then height).
# HF's SmolVLM processors resize through PIL (transformers 4.x and the
# `PilBackend`), so reproducing that integer path is what makes the pixels
# match to the last u8. Keeping the two passes in float gets within 1-2 u8,
# which is enough to flip a greedy token on a near tie.
_PIL_PRECISION_BITS = 22
_PIL_LANCZOS_SUPPORT = 3.0


def _pil_lanczos(x):
    """`lanczos_filter` from Pillow's `Resample.c` (truncated sinc, a=3)."""
    if -3.0 <= x < 3.0:
        if x == 0.0:
            return 1.0
        y = x * math.pi
        z = y / 3.0
        return (math.sin(y) / y) * (math.sin(z) / z)
    return 0.0


def _pil_lanczos_coeffs(in_size, out_size):
    """Port of Pillow's `precompute_coeffs` + `normalize_coeffs_8bpc`.

    Returns `(indices, weights)`, both `(out_size, ksize)` int32 arrays.
    `weights` are the fixed-point taps; `indices` are the source pixels they
    multiply. Taps past the image edge are dropped before normalization,
    exactly as Pillow does, and padded with zero weight (their index is
    clamped only so the gather stays in bounds).
    """
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = _PIL_LANCZOS_SUPPORT * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    inv_filterscale = 1.0 / filterscale
    one = 1 << _PIL_PRECISION_BITS

    indices = np.zeros((out_size, ksize), dtype=np.int64)
    weights = np.zeros((out_size, ksize), dtype=np.int64)
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        # C truncates toward zero; so does `int()`.
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        taps = [
            _pil_lanczos((x + xmin - center + 0.5) * inv_filterscale)
            for x in range(xmax)
        ]
        total = 0.0
        for w in taps:
            total += w
        if total != 0.0:
            taps = [w / total for w in taps]
        for x, w in enumerate(taps):
            # Round half away from zero, as `normalize_coeffs_8bpc` does.
            weights[xx, x] = (
                int(-0.5 + w * one) if w < 0 else int(0.5 + w * one)
            )
            indices[xx, x] = xmin + x
        indices[xx, xmax:] = min(xmin + max(xmax - 1, 0), in_size - 1)
    return indices.astype("int32"), weights.astype("int32")


def _pil_resample_pass(images, in_size, out_size, axis, xp=ops):
    """One fixed-point Lanczos pass of Pillow's resample along `axis`.

    `images` is an int32 `(batch, height, width, channels)` tensor holding
    uint8 values. `xp` supplies `take`, `reshape`, `sum`, `floor_divide` and
    `clip` with `keras.ops` semantics: `keras.ops` itself eagerly, or
    `_TFResampleOps` inside a `tf.data` graph. Returns int32 uint8 values.
    """
    indices, weights = _pil_lanczos_coeffs(in_size, out_size)
    ksize = weights.shape[1]
    gathered = xp.take(images, indices, axis=axis)
    shape = [1, 1, 1, 1, 1]
    shape[axis] = out_size
    shape[axis + 1] = ksize
    acc = xp.sum(gathered * xp.reshape(weights, shape), axis=axis + 1)
    # `clip8((ss + (1 << (PRECISION_BITS - 1))) >> PRECISION_BITS)`: an
    # arithmetic shift floors, which is what `floor_divide` does.
    half = 1 << (_PIL_PRECISION_BITS - 1)
    acc = xp.floor_divide(acc + half, 1 << _PIL_PRECISION_BITS)
    return xp.clip(acc, 0, 255)


class _TFResampleOps:
    """The `keras.ops` names `_pil_resample_pass` needs, on raw `tf`."""

    @staticmethod
    def take(x, indices, axis):
        return tf.gather(x, indices, axis=axis)

    @staticmethod
    def reshape(x, shape):
        return tf.reshape(x, shape)

    @staticmethod
    def sum(x, axis):
        return tf.reduce_sum(x, axis=axis)

    @staticmethod
    def floor_divide(x, y):
        return tf.math.floordiv(x, y)

    @staticmethod
    def clip(x, lo, hi):
        return tf.clip_by_value(x, lo, hi)

    @staticmethod
    def cast(x, dtype):
        return tf.cast(x, dtype)

    @staticmethod
    def round(x):
        return tf.round(x)

    @staticmethod
    def concatenate(xs, axis):
        return tf.concat(xs, axis=axis)


@keras_hub_export("keras_hub.layers.SmolVLM2ImageConverter")
class SmolVLM2ImageConverter(ImageConverter):
    """Image converter for SmolVLM2 models.

    This layer processes images through the same pipeline as HuggingFace's
    ``SmolVLMImageProcessor``:

    1. Resize so the longest edge matches ``size`` (default 2048).
    2. Snap to multiples of ``max_image_size`` (default 512).
    3. Split into sub-image crops + a global view.
    4. Rescale and normalize.

    The output is a dict with:
    - ``"pixel_values"``: float32 tensor of shape
      ``(num_sub_images, max_image_size, max_image_size, 3)``.
    - ``"rows"``: int. Number of rows of sub-image crops.
    - ``"cols"``: int. Number of columns of sub-image crops.

    Args:
        max_image_size: int. Side length of each sub-image crop. Default
            512 (from HF's ``max_image_size.longest_edge``).
        size: int. The longest edge is resized to this before splitting.
            Default 2048 (from HF's ``size.longest_edge``).
        do_image_splitting: bool. Whether to split into sub-images.
            Set ``False`` for video frames. Default ``True``.
        interpolation: str. Resize filter. Default ``"lanczos3"``, which
            matches HF's PIL LANCZOS. ``"lanczos5"`` and the filters
            `ops.image.resize` accepts are also supported.
        antialias: bool. Whether to antialias when downsampling. Default
            ``True``.
    """

    backbone_cls = SmolVLM2Backbone

    def __init__(
        self,
        max_image_size=512,
        size=2048,
        do_image_splitting=True,
        interpolation="lanczos3",
        antialias=True,
        **kwargs,
    ):
        super().__init__(
            interpolation=interpolation, antialias=antialias, **kwargs
        )
        self._check_image_size_unset()
        self.max_image_size = max_image_size
        self.size = size
        self.do_image_splitting = do_image_splitting

    def _check_image_size_unset(self):
        # The crops are sized by `size` and `max_image_size`; the base
        # resize to `image_size` would distort them. The base class lets
        # `image_size` be set after construction, so this also runs per call.
        if self.image_size is not None:
            raise ValueError(
                "`SmolVLM2ImageConverter` does not support `image_size`. Use "
                "`size` and `max_image_size` instead. "
                f"Received: image_size={self.image_size}"
            )

    @preprocessing_function
    def _call_tf(self, inputs):
        if self.do_image_splitting:
            raise ValueError(
                "Training with `do_image_splitting=True` is not "
                "supported: the number of crops depends on each image's "
                "original size, which cannot be resolved inside the "
                "`tf.data` graph. Set `do_image_splitting=False` on the "
                "image converter."
            )

        rank = len(inputs.shape)
        if rank not in (3, 4):
            raise ValueError(
                "`SmolVLM2ImageConverter` expects a single image of shape "
                "`(height, width, channels)` or a batch of shape "
                "`(batch_size, height, width, channels)`. Received inputs "
                f"with shape {tuple(inputs.shape)}."
            )

        images = tf.cast(inputs, "float32")
        if rank == 3:
            images = tf.expand_dims(images, 0)

        ms = self.max_image_size
        h_static, w_static = images.shape[1], images.shape[2]
        is_static = h_static is not None and w_static is not None
        if is_static:
            new_hw = self._sizes_for(h_static, w_static)
        else:
            new_hw = self._sizes_for_dynamic(
                tf.shape(images)[1], tf.shape(images)[2]
            )
        if is_static and self.interpolation == "lanczos3":
            # Same two sizing steps and the same Pillow-exact resample as
            # `_convert_image`, so a `tf.data` training batch gets the same
            # pixels as `generate()`.
            images = self._resize_pil_exact(
                images, (h_static, w_static), new_hw, xp=_TFResampleOps
            )
            images = self._resize_pil_exact(
                images, new_hw, (ms, ms), xp=_TFResampleOps
            )
        else:
            # Other interpolations, or a dynamic height/width (the exact
            # path needs static sizes to build its filter taps): the
            # two-pass `tf.image.resize` pipeline in `_resize_tf`.
            images = _resize_tf(
                images, new_hw, self.interpolation, self.antialias
            )
            images = _resize_tf(
                images, (ms, ms), self.interpolation, self.antialias
            )

        if rank == 3:
            images = tf.squeeze(images, 0)

        pixel_values = self._rescale_and_normalize(images)
        return {
            "pixel_values": pixel_values,
            "rows": tf.constant(0, dtype="int32"),
            "cols": tf.constant(0, dtype="int32"),
        }

    def _static_hw(self, shape):
        """Return the static `(height, width)` from an unbatched image shape."""
        if shape[0] is None or shape[1] is None:
            raise ValueError(
                "`SmolVLM2ImageConverter` needs static image height and "
                "width to compute the sub-image grid. Received an image "
                f"with shape {tuple(shape)}. If you are mapping this layer "
                "over a `tf.data.Dataset`, resize or `set_shape` the images "
                "to a known size first."
            )
        return int(shape[0]), int(shape[1])

    def _sizes_for(self, height, width):
        """HF's two sizing steps: longest edge to `size`, then cap at 4096.

        Every path (`_convert_image`, `_resize_batch`, `_call_tf`) goes
        through this so a training batch and `generate()` see the same
        intermediate size.
        """
        new_h, new_w = _resize_output_size_rescale_to_max_len(
            height, width, max_len=self.size
        )
        return _resize_output_size_scale_below_upper_bound(
            new_h, new_w, max_len=4096
        )

    def _sizes_for_dynamic(self, height, width):
        """`_sizes_for` on symbolic `tf` height/width (a `tf.data` graph).

        Mirrors `_resize_output_size_rescale_to_max_len` and
        `_resize_output_size_scale_below_upper_bound` step for step,
        including the odd-to-even bump and the `int()` truncation, so a
        batch with unknown static size lands on the same intermediate size
        as one with a known size.
        """
        h = tf.cast(height, "float64")
        w = tf.cast(width, "float64")
        aspect = w / h
        wide = w >= h
        max_len = tf.constant(float(self.size), "float64")
        # Step 1: longest edge to `size`; the derived edge is truncated and
        # bumped to even.
        new_w = tf.where(wide, max_len, tf.math.floor(max_len * aspect))
        new_h = tf.where(wide, tf.math.floor(max_len / aspect), max_len)
        new_h = tf.where(
            wide & (tf.math.floormod(new_h, 2.0) != 0.0), new_h + 1.0, new_h
        )
        new_w = tf.where(
            (~wide) & (tf.math.floormod(new_w, 2.0) != 0.0), new_w + 1.0, new_w
        )
        new_h = tf.maximum(new_h, 1.0)
        new_w = tf.maximum(new_w, 1.0)
        # Step 2: cap the longer edge at 4096, truncating the other.
        cap = tf.constant(4096.0, "float64")
        aspect = new_w / new_h
        wide = new_w >= new_h
        cap_w = wide & (new_w > cap)
        cap_h = (~wide) & (new_h > cap)
        new_h = tf.where(cap_w, tf.math.floor(cap / aspect), new_h)
        new_w = tf.where(cap_w, cap, new_w)
        new_w = tf.where(cap_h, tf.math.floor(cap * aspect), new_w)
        new_h = tf.where(cap_h, cap, new_h)
        new_h = tf.cast(tf.maximum(new_h, 1.0), "int32")
        new_w = tf.cast(tf.maximum(new_w, 1.0), "int32")
        return new_h, new_w

    def _call_python(self, inputs):
        outputs = None
        if isinstance(inputs, (list, tuple)):
            if self.do_image_splitting:
                # Each image yields its own number of crops, so convert
                # them one at a time even when they share a size.
                outputs = self._convert_ragged(inputs)
            else:
                try:
                    inputs = np.array(inputs)
                except (ValueError, TypeError, RuntimeError):
                    # Images of different sizes, or backend tensors NumPy
                    # cannot stack (torch tensors on a GPU): convert one at
                    # a time.
                    outputs = self._convert_ragged(inputs)
        if outputs is None:
            outputs = self._convert_image(inputs)
        if in_grain_data_pipeline():
            # Grain pickles outputs across worker processes, so return NumPy
            # arrays rather than backend tensors.
            return convert_preprocessing_outputs_grain(outputs)
        return outputs

    def _convert_ragged(self, images):
        """Convert images of different sizes one at a time.

        With splitting, each image yields its own number of crops, so a list
        of dicts is returned. Without it every image becomes one crop, so
        the crops are stacked into one dict, as for a rank-4 batch.
        """
        outputs = [self._convert_image(image) for image in images]
        if self.do_image_splitting:
            return outputs
        return {
            "pixel_values": ops.concatenate(
                [output["pixel_values"] for output in outputs], axis=0
            ),
            "rows": outputs[0]["rows"],
            "cols": outputs[0]["cols"],
        }

    def _convert_image(self, inputs):
        """Process an image into sub-image crops.

        Args:
            inputs: uint8 or float32 tensor with pixel values in
                `[0, 255]`. Either a single image `(H, W, 3)`, or a batch
                of images `(B, H, W, 3)`. Batched inputs are only
                supported when `do_image_splitting=False`, since the
                number of crops is image dependent.

        Returns:
            dict with `"pixel_values"` `(N, max_image_size,
            max_image_size, 3)`, `"rows"` int, `"cols"` int.
        """
        rank = len(inputs.shape)
        if rank == 4:
            if self.do_image_splitting:
                raise ValueError(
                    "`SmolVLM2ImageConverter` cannot process a batch of "
                    "images with `do_image_splitting=True`, because each "
                    "image yields a different number of crops. Call the "
                    "converter once per image, or set "
                    "`do_image_splitting=False`. "
                    f"Received inputs with shape {tuple(inputs.shape)}."
                )
            return self._resize_batch(inputs)
        if rank != 3:
            raise ValueError(
                "`SmolVLM2ImageConverter` expects a single image of shape "
                "`(height, width, channels)` or a batch of shape "
                "`(batch_size, height, width, channels)`. Received inputs "
                f"with shape {tuple(inputs.shape)}."
            )

        image = ops.cast(inputs, "float32")
        h, w = self._static_hw(image.shape)

        # Step 1: Resize so longest edge = self.size.
        new_h, new_w = self._sizes_for(h, w)

        image = self._resize(ops.expand_dims(image, 0), (new_h, new_w))[0]

        ms = self.max_image_size

        if self.do_image_splitting:
            # Step 2: Snap to multiples of max_image_size.
            aspect_ratio = new_w / new_h
            if new_w >= new_h:
                snap_w = math.ceil(new_w / ms) * ms
                snap_h = int(snap_w / aspect_ratio)
                snap_h = math.ceil(snap_h / ms) * ms
            else:
                snap_h = math.ceil(new_h / ms) * ms
                snap_w = int(snap_h * aspect_ratio)
                snap_w = math.ceil(snap_w / ms) * ms

            image = self._resize(ops.expand_dims(image, 0), (snap_h, snap_w))[0]

            num_rows = 0
            num_cols = 0
            if snap_h > ms or snap_w > ms:
                num_rows = math.ceil(snap_h / ms)
                num_cols = math.ceil(snap_w / ms)

                # Step 3: Split into crops using ops slicing.
                crops = []
                for r in range(num_rows):
                    for c in range(num_cols):
                        crop = image[
                            r * ms : (r + 1) * ms,
                            c * ms : (c + 1) * ms,
                            :,
                        ]
                        crops.append(crop)

                # Global view resized to (ms, ms).
                global_view = self._resize(ops.expand_dims(image, 0), (ms, ms))
                crops.append(global_view[0])

                # Stack: (num_sub_images, ms, ms, 3)
                pixel_values = ops.stack(crops, axis=0)
                pixel_values = ops.cast(pixel_values, "float32")
            else:
                # Image fits in a single crop.
                num_rows = 0
                num_cols = 0
                pixel_values = self._resize(ops.expand_dims(image, 0), (ms, ms))
        else:
            # No splitting (video frames): just resize to square.
            num_rows = 0
            num_cols = 0
            pixel_values = self._resize(ops.expand_dims(image, 0), (ms, ms))

        return {
            "pixel_values": self._rescale_and_normalize(pixel_values),
            "rows": ops.convert_to_tensor(num_rows, dtype="int32"),
            "cols": ops.convert_to_tensor(num_cols, dtype="int32"),
        }

    def _resize(self, images, size):
        """Resize a `(batch, height, width, channels)` batch and clip.

        With `interpolation="lanczos3"` (HF's `resample=LANCZOS`) this
        reproduces Pillow's fixed-point resample bit for bit, see
        `_pil_lanczos_coeffs`. Other interpolations keep the two-pass float
        pipeline (width, clip + round, height, clip + round). `"lanczos5"`
        goes through `scale_and_translate` because `ops.image.resize`
        rejects it on the torch backend (keras-team/keras#23783); the rest
        use `ops.image.resize`.
        """
        in_h, in_w = self._static_hw(images.shape[1:])
        out_h, out_w = size
        if self.interpolation == "lanczos3":
            return self._resize_pil_exact(images, (in_h, in_w), size)
        batch = images.shape[0]
        if batch is None:
            batch = ops.shape(images)[0]

        if self.interpolation == "lanczos5":
            # W pass
            images = ops.image.scale_and_translate(
                images,
                output_shape=(batch, in_h, out_w, images.shape[-1]),
                scale=(1.0, out_w / in_w),
                translation=(0.0, 0.0),
                spatial_dims=(1, 2),
                method=self.interpolation,
                antialias=self.antialias,
            )
            images = ops.round(ops.clip(images, 0.0, 255.0))
            # H pass
            images = ops.image.scale_and_translate(
                images,
                output_shape=(batch, out_h, out_w, images.shape[-1]),
                scale=(out_h / in_h, 1.0),
                translation=(0.0, 0.0),
                spatial_dims=(1, 2),
                method=self.interpolation,
                antialias=self.antialias,
            )
            images = ops.round(ops.clip(images, 0.0, 255.0))
        else:
            # W pass
            images = ops.image.resize(
                images,
                size=(in_h, out_w),
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
            images = ops.round(ops.clip(images, 0.0, 255.0))
            # H pass
            images = ops.image.resize(
                images,
                size=(out_h, out_w),
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
            images = ops.round(ops.clip(images, 0.0, 255.0))
        return images

    @staticmethod
    def _resize_pil_exact(images, in_hw, out_hw, xp=ops):
        """Pillow `Image.resize(..., LANCZOS)` on uint8 values, exactly.

        Width pass first, then height, each clipped to uint8 like Pillow's
        intermediate buffer. `images` is a float tensor of whole uint8
        values; the result is float32 uint8 values. `xp` is `keras.ops` or
        `_TFResampleOps`.
        """
        batch = images.shape[0]
        if batch is not None and batch > 1:
            # One image at a time: the gather in `_pil_resample_pass`
            # materialises `(batch, H, out_w, ksize, C)` int32, about
            # 240 MB per image at the 2048 -> 512 stage, so the peak must
            # not scale with the batch.
            return xp.concatenate(
                [
                    SmolVLM2ImageConverter._resize_pil_exact(
                        images[i : i + 1], in_hw, out_hw, xp=xp
                    )
                    for i in range(batch)
                ],
                axis=0,
            )
        in_h, in_w = in_hw
        out_h, out_w = out_hw
        images = xp.cast(xp.round(images), "int32")
        if out_w != in_w:
            images = _pil_resample_pass(images, in_w, out_w, axis=2, xp=xp)
        if out_h != in_h:
            images = _pil_resample_pass(images, in_h, out_h, axis=1, xp=xp)
        return xp.cast(images, "float32")

    def _rescale_and_normalize(self, pixel_values):
        """Apply the HF `rescale_factor` and mean/std normalization.

        With `image_size` unset, the base `_call_python` skips resizing and
        only applies `scale` and `offset`, in the compute dtype and on the
        image's device. Inside a Grain pipeline the base returns NumPy.
        """
        self._check_image_size_unset()
        return super()._call_python(pixel_values)

    def _resize_batch(self, images):
        """Resize a batch of images to `(max_image_size, max_image_size)`.

        Used when `do_image_splitting=False`, where every image maps to
        exactly one sub-image. Keeps the batch axis, so video frames and
        multimodal training batches can be processed in one call.
        """
        ms = self.max_image_size
        images = ops.cast(images, "float32")
        h, w = self._static_hw(images.shape[1:])
        # Longest-edge step first, as `_convert_image` does, so a training
        # batch gets the same pixels as per-image `generate()` processing.
        images = self._resize(images, self._sizes_for(h, w))
        pixel_values = self._resize(images, (ms, ms))
        return {
            "pixel_values": self._rescale_and_normalize(pixel_values),
            "rows": ops.convert_to_tensor(0, dtype="int32"),
            "cols": ops.convert_to_tensor(0, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_image_size": self.max_image_size,
                "size": self.size,
                "do_image_splitting": self.do_image_splitting,
            }
        )
        return config
