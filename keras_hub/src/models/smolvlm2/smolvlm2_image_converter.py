import math

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs_grain
from keras_hub.src.utils.tensor_utils import in_grain_data_pipeline
from keras_hub.src.utils.tensor_utils import preprocessing_function


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
                except ValueError:
                    # Images of different sizes: convert one at a time.
                    outputs = self._convert_ragged(inputs)
        if outputs is None:
            outputs = self._convert_image(inputs)
        if in_grain_data_pipeline():
            # Grain pickles outputs across worker processes, so return NumPy
            # arrays rather than backend tensors.
            return convert_preprocessing_outputs_grain(outputs)
        return outputs

    @preprocessing_function
    def _call_tf(self, inputs):
        return self._call_python(inputs)

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
        new_h, new_w = _resize_output_size_rescale_to_max_len(
            h, w, max_len=self.size
        )
        new_h, new_w = _resize_output_size_scale_below_upper_bound(
            new_h, new_w, max_len=4096
        )

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

        `ops.image.resize` rejects `"lanczos3"` and `"lanczos5"` on the
        torch backend (keras-team/keras#23783). keras-team/keras#23792
        fixes that by routing both through `scale_and_translate` with
        `scale=out/in` and no translation; this does the same here so the
        output matches on every backend and every Keras version. Once a
        Keras release includes that PR, this branch can collapse into the
        plain `ops.image.resize` call below.
        """
        if self.interpolation in ("lanczos3", "lanczos5"):
            in_h, in_w = self._static_hw(images.shape[1:])
            out_h, out_w = size
            batch = images.shape[0]
            if batch is None:
                batch = ops.shape(images)[0]
            images = ops.image.scale_and_translate(
                images,
                output_shape=(batch, out_h, out_w, images.shape[-1]),
                scale=(out_h / in_h, out_w / in_w),
                translation=(0.0, 0.0),
                spatial_dims=(1, 2),
                method=self.interpolation,
                antialias=self.antialias,
            )
        else:
            images = ops.image.resize(
                images,
                size=size,
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
        return ops.clip(images, 0.0, 255.0)

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
