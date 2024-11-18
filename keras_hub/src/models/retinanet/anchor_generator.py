import math

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export

# TODO: https://github.com/keras-team/keras-hub/issues/1965
from keras_hub.src.bounding_box.converters import convert_format


@keras_hub_export("keras_hub.layers.AnchorGenerator")
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
        self.num_base_anchors = num_scales * len(aspect_ratios)
        self.built = True

    def call(self, inputs):
        images_shape = ops.shape(inputs)
        if len(images_shape) == 4:
            image_shape = images_shape[1:-1]
        else:
            image_shape = images_shape[:-1]

        image_shape = tuple(image_shape)

        multilevel_anchors = {}
        for level in range(self.min_level, self.max_level + 1):
            # Calculate the feature map size for this level
            feat_size_y = math.ceil(image_shape[0] / 2**level)
            feat_size_x = math.ceil(image_shape[1] / 2**level)

            # Calculate the stride (step size) for this level
            stride_y = image_shape[0] // feat_size_y
            stride_x = image_shape[1] // feat_size_x

            # Generate anchor center points
            # Start from stride/2 to center anchors on pixels
            cx = ops.arange(0, feat_size_x, dtype="float32") * stride_x
            cy = ops.arange(0, feat_size_y, dtype="float32") * stride_y

            # Create a grid of anchor centers
            cy_grid, cx_grid = ops.meshgrid(cy, cx, indexing="ij")
            cy_grid = ops.reshape(cy_grid, (-1,))
            cx_grid = ops.reshape(cx_grid, (-1,))

            shifts = ops.stack((cx_grid, cy_grid, cx_grid, cy_grid), axis=1)
            sizes = [
                int(
                    2**level * self.anchor_size * 2 ** (scale / self.num_scales)
                )
                for scale in range(self.num_scales)
            ]

            base_anchors = self.generate_base_anchors(
                sizes=sizes, aspect_ratios=self.aspect_ratios
            )
            shifts = ops.reshape(shifts, (-1, 1, 4))
            base_anchors = ops.reshape(base_anchors, (1, -1, 4))

            anchors = shifts + base_anchors
            anchors = ops.reshape(anchors, (-1, 4))
            multilevel_anchors[f"P{level}"] = convert_format(
                anchors,
                source="xyxy",
                target=self.bounding_box_format,
            )
        return multilevel_anchors

    def generate_base_anchors(self, sizes, aspect_ratios):
        sizes = ops.convert_to_tensor(sizes, dtype="float32")
        aspect_ratios = ops.convert_to_tensor(aspect_ratios)
        h_ratios = ops.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = ops.reshape(w_ratios[:, None] * sizes[None, :], (-1,))
        hs = ops.reshape(h_ratios[:, None] * sizes[None, :], (-1,))

        base_anchors = ops.stack([-1 * ws, -1 * hs, ws, hs], axis=1) / 2
        base_anchors = ops.round(base_anchors)
        return base_anchors

    def compute_output_shape(self, input_shape):
        multilevel_boxes_shape = {}
        if len(input_shape) == 4:
            image_height, image_width = input_shape[1:-1]
        else:
            image_height, image_width = input_shape[:-1]

        for i in range(self.min_level, self.max_level + 1):
            multilevel_boxes_shape[f"P{i}"] = (
                int(
                    math.ceil(image_height / 2 ** (i))
                    * math.ceil(image_width // 2 ** (i))
                    * self.num_base_anchors
                ),
                4,
            )
        return multilevel_boxes_shape
