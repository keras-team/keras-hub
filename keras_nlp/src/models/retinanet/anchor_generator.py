# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import math

import keras
from keras import ops

from keras_nlp.src.bounding_box.converters import convert_format


class Anchor(keras.layers.Layer):
    """A layer used for anchor-based object detectors.

    Args:
        min_level: integer number of minimum level of the output feature pyramid.
        max_level: integer number of maximum level of the output feature pyramid.
        num_scales: integer number representing intermediate scales added on each
            level. For instances, num_scales=2 adds one additional intermediate
            anchor scales [2^0, 2^0.5] on each level.
        aspect_ratios: list of float numbers representing the aspect ratio anchors
            added on each level. The number indicates the ratio of width to height.
            For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
            scale level.
        anchor_size: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
    Call arguments:
        image: an image with shape `[H, W, C]`
        image_shape: a list of integer numbers or Tensors representing [height,
            width] of the input image size.

    Returns:
        multilevel_boxes: an OrderedDict from level to the generated anchor boxes of
        shape `(H/strides[level] * W/strides[level] * len(scales) * len(aspect_ratios), 4)`.

    Example:
    ```python
    anchor_generator = Anchor(
        min_level=3,
        max_level=4,
        num_scales=2,
        aspect_ratios=[0.5, 1., 2.],
        anchor_size=4.,
    )
    anchor_generator(image_shape=(256, 256))
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

    def call(self, image_shape):
        if len(image_shape) != 2:
            raise ValueError(
                "Expected `image_shape` to be a Tensor of rank 2. Got "
                f"image_shape rank={len(image_shape)}"
            )
        image_shape = tuple(image_shape)

        multilevel_boxes = collections.OrderedDict()
        for level in range(self.min_level, self.max_level + 1):
            boxes_l = []
            feat_size_y = math.ceil(image_shape[0] / 2**level)
            feat_size_x = math.ceil(image_shape[1] / 2**level)

            stride_y = ops.cast(image_shape[0] / feat_size_y, "float32")
            stride_x = ops.cast(image_shape[1] / feat_size_x, "float32")

            x = ops.arange(stride_x / 2, image_shape[1], stride_x)
            y = ops.arange(stride_y / 2, image_shape[0], stride_y)

            xv, yv = ops.meshgrid(x, y)

            for scale in range(self.num_scales):
                for aspect_ratio in self.aspect_ratios:
                    intermidate_scale = 2 ** (scale / self.num_scales)
                    base_anchor_size = (
                        self.anchor_size * 2**level * intermidate_scale
                    )
                    aspect_x = aspect_ratio**0.5
                    aspect_y = aspect_ratio**-0.5
                    half_anchor_size_x = base_anchor_size * aspect_x / 2.0
                    half_anchor_size_y = base_anchor_size * aspect_y / 2.0

                    # Tensor shape Nx4
                    boxes = ops.stack(
                        [
                            yv - half_anchor_size_y,
                            xv - half_anchor_size_x,
                            yv + half_anchor_size_y,
                            xv + half_anchor_size_x,
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
        for level in range(self.min_level, self.max_level + 1):
            multilevel_boxes_shape[f"P{level}"] = (None, None, 4)
        return multilevel_boxes_shape

    @property
    def anchors_per_location(self):
        """
        anchors_per_location: number of anchors per pixel location.
        """
        return self.num_scales * len(self.aspect_ratios)
