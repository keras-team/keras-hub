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
    """Projector layer for Qwen2-VL."""

    def __init__(self, hidden_size, output_hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_hidden_size = output_hidden_size

        self.merger = keras.Sequential(
            [
                layers.InputLayer(
                    input_shape=(4 * hidden_size,), name="merger_input"
                ),
                layers.Dense(output_hidden_size, name="merger_proj"),
                layers.Activation("gelu", name="activation"),
                layers.Dense(output_hidden_size, name="output_proj"),
            ],
            name="merger",
        )

    def call(self, x):
        # Use ops.shape() to get symbolic tensors
        input_shape = ops.shape(x)
        B = input_shape[0]
        H, W, C = input_shape[2], input_shape[3], input_shape[4]

        # 1. Fold Time into Batch: (-1, H, W, C)
        x = ops.reshape(x, (-1, H, W, C))

        # 2. Reshape to isolate 2x2 blocks
        # Shape: (-1, H/2, 2, W/2, 2, C)
        x = ops.reshape(x, (-1, H // 2, 2, W // 2, 2, C))

        # 3. Permute to bring the 2x2 blocks together
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))

        # 4. Flatten the 2x2 blocks
        # Shape: (-1, H/2, W/2, 4*C)
        x = ops.reshape(x, (-1, H // 2, W // 2, 4 * C))

        # 5. CRITICAL FIX: Flatten to 2D (Batch*Tokens, Features)
        # The Sequential 'merger' layer demands Rank 2 input.
        x = ops.reshape(x, (-1, 4 * C))

        # 6. MLP Projection
        x = self.merger(x)

        # 7. Restore Batch Dimension and Flatten Sequence
        # (Batch, Seq_Len, Output_Hidden)
        x = ops.reshape(x, (B, -1, self.output_hidden_size))

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, self.output_hidden_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "output_hidden_size": self.output_hidden_size,
            }
        )
        return config
