# Copyright 2024 The KerasHub Authors
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
import math

import numpy as np
from keras import layers
from keras import ops


@keras_hub_export("keras_hub.layers.Qwen2VLImageConverter")
class Qwen2VLImageConverter(layers.Layer):
    """Image converter for Qwen2-VL that handles smart resizing and normalization.

    This layer analyzes the aspect ratio of input images and resizes them
    to an optimal grid size that is a multiple of the patch size.

    Args:
        min_pixels: Int. Minimum number of pixels for the resized image.
        max_pixels: Int. Maximum number of pixels for the resized image.
        patch_size: Int. The patch size of the vision encoder (default 14).
        mean: List/Tuple. Mean values for normalization.
        std: List/Tuple. Standard deviation values for normalization.
    """

    def __init__(
        self,
        min_pixels=224 * 224,
        max_pixels=1280 * 28 * 28,
        patch_size=14,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.mean = np.array(mean, dtype="float32")
        self.std = np.array(std, dtype="float32")
        self.rescaling_layer = layers.Rescaling(scale=1.0 / 255.0)

    def _smart_resize(self, height, width):
        """Calculates the optimal new dimensions."""
        pixel_count = height * width
        scale = 1.0

        if pixel_count < self.min_pixels:
            scale = math.sqrt(self.min_pixels / pixel_count)
        elif pixel_count > self.max_pixels:
            scale = math.sqrt(self.max_pixels / pixel_count)

        new_h = int(height * scale)
        new_w = int(width * scale)

        # Snap to multiples of 2x patch_size (28)
        snap = self.patch_size * 2
        new_h = round(new_h / snap) * snap
        new_w = round(new_w / snap) * snap

        return new_h, new_w

    def call(self, image):
        input_shape = ops.shape(image)
        h, w = input_shape[-3], input_shape[-2]

        # Note: In graph execution, h/w might be symbolic.
        # Smart resizing logic typically runs on CPU/NumPy side in preprocessing pipelines.
        # For this implementation, we assume values are available or eager execution.
        new_h, new_w = self._smart_resize(float(h), float(w))

        resized_image = ops.image.resize(image, (new_h, new_w))

        # Normalize
        x = self.rescaling_layer(resized_image)
        x = (x - self.mean) / self.std

        # Add Time dimension if missing (static image case)
        if len(ops.shape(x)) == 3:
            x = ops.expand_dims(x, axis=0)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "patch_size": self.patch_size,
            }
        )
        return config
