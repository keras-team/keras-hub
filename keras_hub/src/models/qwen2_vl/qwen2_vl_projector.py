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
import keras
from keras import layers
from keras import ops


class Qwen2VLProjector(layers.Layer):
    """Projector layer for Qwen2-VL.

    This layer downsamples vision features by merging 2x2 neighboring patches
    into a single token and projecting them to the LLM's hidden size.

    Args:
        hidden_size: int. The hidden size of the vision encoder output.
        output_hidden_size: int. The hidden size of the LLM (text model).
        **kwargs: Standard Keras keyword arguments.

    Example:
    ```python
    images = keras.random.ones((1, 28, 28, 1152))
    projector = Qwen2VLProjector(hidden_size=1152, output_hidden_size=4096)
    outputs = projector(images)
    ```
    """

    def __init__(self, hidden_size, output_hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_hidden_size = output_hidden_size

        self.merger = layers.Sequential(
            [
                layers.Dense(output_hidden_size, name="merger_proj"),
                layers.Activation("gelu", name="activation"),
                layers.Dense(output_hidden_size, name="output_proj"),
            ],
            name="merger",
        )

    def call(self, x):
        input_shape = ops.shape(x)
        H, W, C = input_shape[1], input_shape[2], input_shape[3]

        # Reshape to isolate 2x2 blocks: (B, H/2, 2, W/2, 2, C)
        x = ops.reshape(x, (-1, H // 2, 2, W // 2, 2, C))

        # Permute to bring the 2x2 blocks together: (B, H/2, W/2, 2, 2, C)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))

        # Flatten the 2x2xC block into a single vector: (B, H/2, W/2, 4*C)
        x = ops.reshape(x, (-1, H // 2, W // 2, 4 * C))

        x = self.merger(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "output_hidden_size": self.output_hidden_size,
            }
        )
        return config