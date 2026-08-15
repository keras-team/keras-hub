import math

from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
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
    """

    backbone_cls = SmolVLM2Backbone

    def __init__(
        self,
        max_image_size=512,
        size=2048,
        do_image_splitting=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_image_size = max_image_size
        self.size = size
        self.do_image_splitting = do_image_splitting

    @preprocessing_function
    def call(self, inputs):
        """Process a single image into sub-image crops.

        Args:
            inputs: uint8 or float32 tensor ``(H, W, 3)`` with pixel
                values in ``[0, 255]``.
        Returns:
            dict with ``"pixel_values"`` ``(N, max_image_size,
            max_image_size, 3)``, ``"rows"`` int, ``"cols"`` int.
        """
        image = ops.cast(inputs, "float32")
        h = int(ops.shape(image)[0])
        w = int(ops.shape(image)[1])

        # Step 1: Resize so longest edge = self.size.
        new_h, new_w = _resize_output_size_rescale_to_max_len(
            h, w, max_len=self.size
        )
        new_h, new_w = _resize_output_size_scale_below_upper_bound(
            new_h, new_w, max_len=4096
        )

        image = ops.image.resize(
            ops.expand_dims(image, 0),
            size=(new_h, new_w),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )[0]
        image = ops.clip(image, 0.0, 255.0)

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

            image = ops.image.resize(
                ops.expand_dims(image, 0),
                size=(snap_h, snap_w),
                interpolation=self.interpolation,
                antialias=self.antialias,
            )[0]
            image = ops.clip(image, 0.0, 255.0)

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
                global_view = ops.image.resize(
                    ops.expand_dims(image, 0),
                    size=(ms, ms),
                    interpolation=self.interpolation,
                    antialias=self.antialias,
                )[0]
                global_view = ops.clip(global_view, 0.0, 255.0)
                crops.append(global_view)

                # Stack: (num_sub_images, ms, ms, 3)
                pixel_values = ops.stack(crops, axis=0)
                pixel_values = ops.cast(pixel_values, "float32")
            else:
                # Image fits in a single crop.
                num_rows = 0
                num_cols = 0
                pixel_values = ops.image.resize(
                    ops.expand_dims(image, 0),
                    size=(ms, ms),
                    interpolation=self.interpolation,
                    antialias=self.antialias,
                )
                pixel_values = ops.cast(
                    ops.clip(pixel_values, 0.0, 255.0), "float32"
                )
        else:
            # No splitting (video frames): just resize to square.
            num_rows = 0
            num_cols = 0
            pixel_values = ops.image.resize(
                ops.expand_dims(image, 0),
                size=(ms, ms),
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
            pixel_values = ops.cast(
                ops.clip(pixel_values, 0.0, 255.0), "float32"
            )

        # Step 4: Rescale and normalize.
        if self.scale is not None:
            scale = ops.convert_to_tensor(self.scale, dtype="float32")
            scale = ops.reshape(scale, (1, 1, 1, -1))
            pixel_values = pixel_values * scale
        if self.offset is not None:
            offset = ops.convert_to_tensor(self.offset, dtype="float32")
            offset = ops.reshape(offset, (1, 1, 1, -1))
            pixel_values = pixel_values + offset

        return {
            "pixel_values": pixel_values,
            "rows": ops.convert_to_tensor(num_rows, dtype="int32"),
            "cols": ops.convert_to_tensor(num_cols, dtype="int32"),
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
