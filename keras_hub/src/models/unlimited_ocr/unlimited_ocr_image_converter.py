import functools
import math

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.mistral3.mistral3_image_converter import _resize_axis
from keras_hub.src.models.mistral3.mistral3_image_converter import (
    _resize_axis_tf,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import in_grain_data_pipeline
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


def _tile_grids(min_tiles, max_tiles):
    """Lists the `(columns, rows)` grids, sorted by their number of tiles."""
    grids = []
    for columns in range(1, max_tiles + 1):
        for rows in range(1, max_tiles + 1):
            if min_tiles <= columns * rows <= max_tiles:
                grids.append((columns, rows))
    return sorted(grids, key=lambda grid: grid[0] * grid[1])


@functools.lru_cache(maxsize=128)
def get_optimal_tiled_canvas(image_size, tile_size, min_tiles, max_tiles):
    """Finds the tile grid with the closest aspect ratio.

    The function follows `get_optimal_tiled_canvas` in
    `image_processing_unlimited_ocr.py:102-137`. In a tie, it prefers the grid
    with more tiles while the tiles cover less than twice the image area.

    Args:
        image_size: tuple of two ints. The `(height, width)` of the image.
        tile_size: int. The side of a tile.
        min_tiles: int. The minimum number of tiles.
        max_tiles: int. The maximum number of tiles.

    Returns:
        A `(columns, rows)` tuple of ints.
    """
    height, width = image_size
    aspect_ratio = width / height
    area = width * height
    best_difference = float("inf")
    best_grid = (1, 1)
    for grid in _tile_grids(min_tiles, max_tiles):
        difference = abs(aspect_ratio - grid[0] / grid[1])
        if difference < best_difference:
            best_difference = difference
            best_grid = grid
        elif difference == best_difference:
            if area > 0.5 * tile_size * tile_size * grid[0] * grid[1]:
                best_grid = grid
    return best_grid


def _get_optimal_tiled_canvas_tf(
    height, width, tile_size, min_tiles, max_tiles
):
    """Computes `get_optimal_tiled_canvas` with TensorFlow ops.

    The Python loop keeps the first grid with the smallest difference. A
    later grid with the same difference replaces it when the image area is
    above half of the area of the grid. The function gives the same result
    with a vector test.

    Args:
        height: scalar int tensor. The height of the image.
        width: scalar int tensor. The width of the image.
        tile_size: int. The side of a tile.
        min_tiles: int. The minimum number of tiles.
        max_tiles: int. The maximum number of tiles.

    Returns:
        A `(columns, rows)` tuple of scalar `int32` tensors.
    """
    grids = np.array(_tile_grids(min_tiles, max_tiles), dtype="float64")
    columns = tf.constant(grids[:, 0])
    rows = tf.constant(grids[:, 1])
    height = tf.cast(height, "float64")
    width = tf.cast(width, "float64")
    difference = tf.abs(width / height - columns / rows)
    tied = difference == tf.reduce_min(difference)
    first = tied & (tf.cumsum(tf.cast(tied, "int32")) == 1)
    larger = width * height > 0.5 * tile_size * tile_size * columns * rows
    eligible = first | (tied & larger)
    indices = tf.range(grids.shape[0])
    best = tf.reduce_max(tf.where(eligible, indices, -1))
    return tf.cast(columns[best], "int32"), tf.cast(rows[best], "int32")


def _resize_uint8(image, height, width):
    """Resizes a `(H, W, C)` image with integer-valued pixels in `[0, 255]`.

    Hugging Face resizes the `uint8` image with torchvision. The same integer
    passes reproduce its output.
    """
    image = _resize_axis(image.astype("float64"), 1, width)
    return _resize_axis(image, 0, height)


@keras_hub_export("keras_hub.layers.UnlimitedOCRImageConverter")
class UnlimitedOCRImageConverter(ImageConverter):
    """Converts raw images into `UnlimitedOCRBackbone` image inputs.

    Each image gives one global view and a number of local tiles.

    The global view keeps the aspect ratio of the image. The converter
    resizes the long side to `global_size` and pads the image to a square
    with `background_color`. The local tiles exist only for an image with a
    side longer than `tile_size` when `crop_to_patches` is `True`. The
    converter picks a grid of `columns * rows` tiles between `min_tiles` and
    `max_tiles` with the closest aspect ratio. It resizes the image to the
    full grid size without keeping the aspect ratio and cuts it into tiles
    in row-major order. Then the converter rescales and normalizes all
    pixels.

    The layer takes a list of `(height, width, 3)` images, or an array of
    images with a batch axis. A single `(height, width, 3)` image counts as a
    batch of one image. The layer outputs a dictionary:

    - `"pixel_values"`: float array of shape
        `(num_images, global_size, global_size, 3)`.
    - `"pixel_values_local"`: float array of shape
        `(num_tiles, tile_size, tile_size, 3)`. The array holds the tiles of
        all images one after the other.
    - `"patches_grid"`: int array of shape `(num_images, 2)`. The columns and
        rows of tiles of each image. An image with no tiles has `(0, 0)`.

    Args:
        global_size: int. The side of the global view in pixels. Defaults to
            `1024`.
        tile_size: int. The side of a local tile in pixels. Defaults to
            `640`.
        min_tiles: int. The minimum number of tiles of an image that gets
            tiles. Defaults to `2`.
        max_tiles: int. The maximum number of tiles of an image. Defaults to
            `32`.
        crop_to_patches: bool. If `False`, no image gets tiles. Defaults to
            `True`.
        background_color: tuple of three ints. The pad color of the global
            view, in pixel units. Defaults to `(127, 127, 127)`.
        maximum_pad_value: int. If `crop_to_patches` is `False` and
            `global_size` is at most this value, the converter resizes the
            image to a `global_size` square and does not keep the aspect
            ratio. Otherwise the converter keeps the aspect ratio and pads.
            Defaults to `640`.
        patch_size: int. The patch size of the vision encoder. Defaults to
            `16`.
        downsample_ratio: int. The downsample ratio after the vision encoder
            patches. Defaults to `4`.
        image_mean: tuple of three floats. The per-channel mean, in `[0, 1]`
            units. Defaults to `(0.5, 0.5, 0.5)`.
        image_std: tuple of three floats. The per-channel standard deviation,
            in `[0, 1]` units. Defaults to `(0.5, 0.5, 0.5)`.
        rescale_factor: float. The factor that maps `[0, 255]` pixels to
            `[0, 1]`. Defaults to `1 / 255`.

    Example:
    ```python
    converter = keras_hub.layers.UnlimitedOCRImageConverter.from_preset(
        "unlimited_ocr"
    )
    outputs = converter([np.random.randint(0, 256, (900, 700, 3))])
    outputs["pixel_values"].shape  # (1, 1024, 1024, 3)
    ```
    """

    backbone_cls = UnlimitedOCRBackbone

    def __init__(
        self,
        global_size=1024,
        tile_size=640,
        min_tiles=2,
        max_tiles=32,
        crop_to_patches=True,
        background_color=(127, 127, 127),
        maximum_pad_value=640,
        patch_size=16,
        downsample_ratio=4,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor=1.0 / 255.0,
        **kwargs,
    ):
        # Drop the base class arguments that this layer fixes or does not
        # use. A config from the base class `get_config` can carry them.
        for key in (
            "scale",
            "offset",
            "dtype",
            "image_size",
            "crop_to_aspect_ratio",
            "pad_to_aspect_ratio",
            "interpolation",
            "antialias",
            "bounding_box_format",
        ):
            kwargs.pop(key, None)
        super().__init__(
            image_size=None,
            crop_to_aspect_ratio=False,
            interpolation="bicubic",
            antialias=True,
            dtype="float32",
            **kwargs,
        )
        self.global_size = global_size
        self.tile_size = tile_size
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.crop_to_patches = crop_to_patches
        self.background_color = tuple(background_color)
        self.maximum_pad_value = maximum_pad_value
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)
        self.rescale_factor = rescale_factor
        # Hugging Face multiplies the `float32` mean and std by
        # `1 / rescale_factor` in `float32`, so the same steps are needed.
        to_pixels = np.float32(1.0 / rescale_factor)
        self._mean_pixels = np.array(image_mean, dtype="float32") * to_pixels
        self._std_pixels = np.array(image_std, dtype="float32") * to_pixels

    @property
    def global_tokens_per_side(self):
        """The side of the grid of global features."""
        side = self.global_size // self.patch_size
        return math.ceil(side / self.downsample_ratio)

    @property
    def local_tokens_per_side(self):
        """The side of the grid of features of one local tile."""
        side = self.tile_size // self.patch_size
        return math.ceil(side / self.downsample_ratio)

    def get_num_image_tokens(self, columns, rows):
        """Returns the number of placeholder tokens of one image.

        Args:
            columns: int. The number of tile columns. Use `0` for no tiles.
            rows: int. The number of tile rows. Use `0` for no tiles.

        Returns:
            The number of image tokens: the local rows, the global rows with
            their newlines, and the view separator.
        """
        side = self.global_tokens_per_side
        count = side * (side + 1) + 1
        if columns * rows > 0:
            local_side = self.local_tokens_per_side
            count += rows * local_side * (columns * local_side + 1)
        return count

    def _get_num_image_tokens_tf(self, columns, rows):
        """Computes `get_num_image_tokens` for int tensors of grids."""
        side = self.global_tokens_per_side
        local_side = self.local_tokens_per_side
        local = rows * local_side * (columns * local_side + 1)
        return side * (side + 1) + 1 + tf.where(columns * rows > 0, local, 0)

    def _normalize(self, image):
        return (image.astype("float32") - self._mean_pixels) / self._std_pixels

    def _convert_one(self, image):
        """Converts one `(height, width, 3)` image with NumPy."""
        image = np.clip(np.round(image.astype("float32")), 0, 255)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "Each image must have shape `(height, width, 3)`. "
                f"Received: {image.shape}"
            )
        height, width = image.shape[:2]

        # Local tiles (image_processing_unlimited_ocr.py:245-260).
        tiles = np.zeros((0, self.tile_size, self.tile_size, 3), "float32")
        grid = (0, 0)
        if self.crop_to_patches and max(height, width) > self.tile_size:
            columns, rows = get_optimal_tiled_canvas(
                (height, width),
                self.tile_size,
                self.min_tiles,
                self.max_tiles,
            )
            resized = _resize_uint8(
                image, self.tile_size * rows, self.tile_size * columns
            )
            tiles = []
            for index in range(columns * rows):
                row, column = divmod(index, columns)
                tile = resized[
                    row * self.tile_size : (row + 1) * self.tile_size,
                    column * self.tile_size : (column + 1) * self.tile_size,
                ]
                tiles.append(self._normalize(tile))
            tiles = np.stack(tiles, axis=0)
            grid = (columns, rows)

        # Global view (image_processing_unlimited_ocr.py:285-290). The view
        # keeps the aspect ratio and pads to a square on raw pixels.
        if (
            not self.crop_to_patches
            and self.global_size <= self.maximum_pad_value
        ):
            new_height = new_width = self.global_size
        else:
            scale = self.global_size / max(height, width)
            new_height = round(height * scale)
            new_width = round(width * scale)
        view = _resize_uint8(image, new_height, new_width)
        if new_height != new_width:
            side = max(new_height, new_width)
            left = (side - new_width) // 2
            top = (side - new_height) // 2
            canvas = np.empty((side, side, 3), dtype="float64")
            canvas[:] = np.array(self.background_color, dtype="float64")
            canvas[top : top + new_height, left : left + new_width] = view
            view = canvas
        return self._normalize(view), tiles, grid

    def _convert_images(self, images):
        views, tiles, grids = [], [], []
        for image in images:
            view, image_tiles, grid = self._convert_one(image)
            views.append(view)
            tiles.append(image_tiles)
            grids.append(grid)
        pixel_values = np.stack(views, axis=0).astype("float32")
        pixel_values_local = np.concatenate(tiles, axis=0).astype("float32")
        patches_grid = np.array(grids, dtype="int32").reshape(-1, 2)
        return pixel_values, pixel_values_local, patches_grid

    def _call_python(self, inputs):
        if isinstance(inputs, (list, tuple)):
            images = [ops.convert_to_numpy(image) for image in inputs]
        else:
            inputs = ops.convert_to_numpy(inputs)
            images = [inputs] if inputs.ndim == 3 else list(inputs)
        pixel_values, pixel_values_local, patches_grid = self._convert_images(
            images
        )
        outputs = {
            "pixel_values": pixel_values,
            "pixel_values_local": pixel_values_local,
            "patches_grid": patches_grid,
        }
        if in_grain_data_pipeline():
            return convert_preprocessing_outputs_python(outputs)
        return {k: ops.convert_to_tensor(v) for k, v in outputs.items()}

    def _normalize_tf(self, image):
        mean = tf.constant(self._mean_pixels, dtype="float32")
        std = tf.constant(self._std_pixels, dtype="float32")
        return (tf.cast(image, "float32") - mean) / std

    def _convert_batch_tf(self, images):
        """Converts `(N, H, W, 3)` images of one size with TensorFlow ops.

        The function follows `_convert_one`. The tile grid, the tile count,
        and the global view size depend on the image size, so the function
        reads the dynamic shape.
        """
        if images.shape[-1] not in (None, 3):
            raise ValueError(
                "Each image must have shape `(height, width, 3)`. "
                f"Received: {images.shape[1:]}"
            )
        images = tf.clip_by_value(tf.round(images), 0.0, 255.0)
        images = tf.cast(images, "float64")
        shape = tf.shape(images)
        batch, height, width = shape[0], shape[1], shape[2]
        tile_size = self.tile_size

        # Local tiles. All images share one size, so they share one grid.
        columns = rows = tf.constant(0, "int32")
        tiles = tf.zeros((0, tile_size, tile_size, 3), "float32")
        if self.crop_to_patches:

            def tiled():
                columns, rows = _get_optimal_tiled_canvas_tf(
                    height,
                    width,
                    tile_size,
                    self.min_tiles,
                    self.max_tiles,
                )
                resized = _resize_axis_tf(images, 2, tile_size * columns)
                resized = _resize_axis_tf(resized, 1, tile_size * rows)
                resized = tf.reshape(
                    resized, (batch, rows, tile_size, columns, tile_size, 3)
                )
                resized = tf.transpose(resized, (0, 1, 3, 2, 4, 5))
                resized = tf.reshape(resized, (-1, tile_size, tile_size, 3))
                return self._normalize_tf(resized), columns, rows

            def untiled():
                zero = tf.constant(0, "int32")
                return tiles, zero, zero

            tiles, columns, rows = tf.cond(
                tf.maximum(height, width) > tile_size, tiled, untiled
            )

        # Global view. The view keeps the aspect ratio and pads to a square
        # on raw pixels.
        if (
            not self.crop_to_patches
            and self.global_size <= self.maximum_pad_value
        ):
            new_height = new_width = tf.constant(self.global_size, "int32")
        else:
            scale = self.global_size / tf.cast(
                tf.maximum(height, width), "float64"
            )
            new_height = tf.cast(
                tf.round(tf.cast(height, "float64") * scale), "int32"
            )
            new_width = tf.cast(
                tf.round(tf.cast(width, "float64") * scale), "int32"
            )
        view = _resize_axis_tf(images, 2, new_width)
        view = _resize_axis_tf(view, 1, new_height)
        side = tf.maximum(new_height, new_width)
        top = (side - new_height) // 2
        left = (side - new_width) // 2
        background = tf.constant(self.background_color, "float64")
        view = tf.pad(
            view - background,
            [
                [0, 0],
                [top, side - new_height - top],
                [left, side - new_width - left],
                [0, 0],
            ],
        )
        view = self._normalize_tf(view + background)

        pixel_values = tf.ensure_shape(
            view, [None, self.global_size, self.global_size, 3]
        )
        pixel_values_local = tf.ensure_shape(
            tiles, [None, tile_size, tile_size, 3]
        )
        patches_grid = tf.tile(tf.stack([columns, rows])[None, :], (batch, 1))
        return pixel_values, pixel_values_local, patches_grid

    @preprocessing_function
    def _call_tf(self, inputs):
        images = tf.cast(inputs, "float32")
        if images.shape.rank == 3:
            images = tf.expand_dims(images, 0)
        pixel_values, pixel_values_local, patches_grid = self._convert_batch_tf(
            images
        )
        return {
            "pixel_values": pixel_values,
            "pixel_values_local": pixel_values_local,
            "patches_grid": patches_grid,
        }

    def get_config(self):
        config = super().get_config()
        for key in (
            "scale",
            "offset",
            "dtype",
            "image_size",
            "crop_to_aspect_ratio",
            "pad_to_aspect_ratio",
            "interpolation",
            "antialias",
            "bounding_box_format",
        ):
            config.pop(key, None)
        config.update(
            {
                "global_size": self.global_size,
                "tile_size": self.tile_size,
                "min_tiles": self.min_tiles,
                "max_tiles": self.max_tiles,
                "crop_to_patches": self.crop_to_patches,
                "background_color": self.background_color,
                "maximum_pad_value": self.maximum_pad_value,
                "patch_size": self.patch_size,
                "downsample_ratio": self.downsample_ratio,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
                "rescale_factor": self.rescale_factor,
            }
        )
        return config
