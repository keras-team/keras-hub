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
from keras import layers
from keras import ops


class Qwen2VLProjector(layers.Layer):
    """Qwen2-VL Projector.

    This layer projects the vision encoder outputs to the text embedding space.
    It reshapes the 5D video/image features into a 2D sequence of tokens.
    """

    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dense = layers.Dense(output_dim, name="dense")

    def call(self, x):
        # x shape: (Batch, Time, Height, Width, Channels)
        shape = ops.shape(x)
        B, _, _, _, C = shape[0], shape[1], shape[2], shape[3], shape[4]

        # Flatten Time, Height, Width into Sequence Length
        # (Batch, T*H*W, Channels)
        x = ops.reshape(x, (B, -1, C))

        # Project to text embedding dimension
        x = self.dense(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            }
        )
        return config
