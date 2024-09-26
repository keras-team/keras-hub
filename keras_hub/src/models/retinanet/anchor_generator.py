import math

import keras
from keras import ops

from keras_hub.src.bounding_box.converters import convert_format


class AnchorGenerator(keras.layers.Layer):
    """Generates anchor boxes for object detection tasks.

    This layer creates a set of anchor boxes (also known as default boxes or
    priors) for use in object detection models, particularly those utilizing
    Feature Pyramid Networks (FPN). It generates anchors across multiple
    pyramid levels, with various scales and aspect ratios.

    Feature Pyramid Levels:
    - Levels typically range from 2 to 6 (P2 to P7), corresponding to different
        resolutions of the input image.
    - Each level l has a stride of 2^l pixels relative to the input image.
    - Lower levels (e.g., P2) have higher resolution and are used for
        detecting smaller objects.
    - Higher levels (e.g., P7) have lower resolution and are used
        for larger objects.

    Args:
        bounding_box_format: str. The format of the bounding boxes
            to be generated. Expected to be a string like 'xyxy', 'xywh', etc.
        min_level: int. Minimum level of the output feature pyramid.
        max_level: int. Maximum level of the output feature pyramid.
        num_scales: int. Number of intermediate scales added on each level.
            For example, num_scales=2 adds one additional intermediate anchor
            scale [2^0, 2^0.5] on each level.
        aspect_ratios:  List[float]. Aspect ratios of anchors added on
            each level. Each number indicates the ratio of width to height.
        anchor_size: float. Scale of size of the base anchor relative to the
            feature stride 2^level.

    Call arguments:
        inputs: An image tensor with shape `[B, H, W, C]` or
            `[H, W, C]`. Its shape will be used to determine anchor
            sizes.

    Returns:
        Dict: A dictionary mapping feature levels
            (e.g., 'P3', 'P4', etc.) to anchor boxes. Each entry contains a
            tensor  of shape
            `(H/stride * W/stride * num_anchors_per_location, 4)`,
            where H and W are the height and width of the image,
            stride is 2^level, and num_anchors_per_location is
            `num_scales * len(aspect_ratios)`.

    Example:
    ```python
    anchor_generator = AnchorGenerator(
        bounding_box_format='xyxy',
        min_level=3,
        max_level=7,
        num_scales=3,
        aspect_ratios=[0.5, 1.0, 2.0],
        anchor_size=4.0,
    )
    anchors = anchor_generator(images=keas.ops.ones(shape=(2, 640, 480, 3)))
    ```
    """

    def __init__(
        self,
        bounding_box_format,
        min_level,
        max_level,
        num_scales,
        aspect_ratios,
        anchor_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_size = anchor_size
        self.built = True

    def call(self, inputs):
        images_shape = ops.shape(inputs)
        if len(images_shape) == 4:
            image_shape = images_shape[1:-1]
        else:
            image_shape = images_shape[:-1]

        image_shape = tuple(image_shape)

        multilevel_boxes = {}
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            # Calculate the feature map size for this level
            feat_size_y = math.ceil(image_shape[0] / 2**level)
            feat_size_x = math.ceil(image_shape[1] / 2**level)

            # Calculate the stride (step size) for this level
            stride_y = ops.cast(image_shape[0] / feat_size_y, "float32")
            stride_x = ops.cast(image_shape[1] / feat_size_x, "float32")

            # Generate anchor center points
            # Start from stride/2 to center anchors on pixels
            cx = ops.arange(stride_x / 2, image_shape[1], stride_x)
            cy = ops.arange(stride_y / 2, image_shape[0], stride_y)

            # Create a grid of anchor centers
            cx_grid, cy_grid = ops.meshgrid(cx, cy)

            for scale in range(self.num_scales):
                for aspect_ratio in self.aspect_ratios:
                    # Calculate the intermediate scale factor
                    intermidate_scale = 2 ** (scale / self.num_scales)
                    # Calculate the base anchor size for this level and scale
                    base_anchor_size = (
                        self.anchor_size * 2**level * intermidate_scale
                    )
                    # Adjust anchor dimensions based on aspect ratio
                    aspect_x = aspect_ratio**0.5
                    aspect_y = aspect_ratio**-0.5
                    half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                    half_anchor_size_y = base_anchor_size * aspect_y / 2.0

                    # Generate anchor boxes (y1, x1, y2, x2 format)
                    boxes = ops.stack(
                        [
                            cy_grid - half_anchor_size_y,
                            cx_grid - half_anchor_size_x,
                            cy_grid + half_anchor_size_y,
                            cx_grid + half_anchor_size_x,
                        ],
                        axis=-1,
                    )
                    boxes_l.append(boxes)
            # Concat anchors on the same level to tensor shape HxWx(Ax4)
            boxes_l = ops.concatenate(boxes_l, axis=-1)
            boxes_l = ops.reshape(boxes_l, (-1, 4))
            # Convert to user defined
            multilevel_boxes[f"P{level}"] = convert_format(
                boxes_l,
                source="yxyx",
                target=self.bounding_box_format,
            )
        return multilevel_boxes

    def compute_output_shape(self, input_shape):
        multilevel_boxes_shape = {}
        if len(input_shape) == 4:
            image_height, image_width = input_shape[1:-1]
        else:
            image_height, image_width = input_shape[:-1]

        for i in range(self.min_level, self.max_level + 1):
            multilevel_boxes_shape[f"P{i}"] = (
                (image_height // 2 ** (i))
                * (image_width // 2 ** (i))
                * self.anchors_per_location,
                4,
            )
        return multilevel_boxes_shape

    @property
    def anchors_per_location(self):
        """
        The `anchors_per_location` property returns the number of anchors
        generated per pixel location, which is equal to
        `num_scales * len(aspect_ratios)`.
        """
        return self.num_scales * len(self.aspect_ratios)
