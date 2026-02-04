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
from keras_hub.src.models.backbone import Backbone


class Qwen2VLVisionEncoder(Backbone):
    """Qwen2-VL Vision Encoder (ViT).

    This class implements the vision encoder for Qwen2-VL, based on the
    Vision Transformer (ViT) architecture.

    Args:
        patch_size: int. The spatial patch size of the images.
        temporal_patch_size: int. The temporal patch size for video inputs.
        hidden_size: int. The hidden size of the transformer layers.
        depth: int. The number of transformer blocks.
        num_heads: int. The number of attention heads.
        mlp_ratio: int. The ratio of the hidden size of the MLP to the
            hidden size of the transformer.
        activation: string. The activation function to use.
        dtype: string or keras.mixed_precision.DTypePolicy. The dtype to use
            for the model computations and weights.
        **kwargs: Standard Keras keyword arguments.

    Example:
    ```python
    encoder = Qwen2VLVisionEncoder(
        patch_size=14,
        temporal_patch_size=2,
        hidden_size=1152,
        depth=2,
        num_heads=16,
    )
    images = keras.random.ones((1, 2, 224, 224, 3))
    outputs = encoder(images)
    ```
    """

    def __init__(
        self,
        patch_size=14,
        temporal_patch_size=2,
        hidden_size=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=4,
        activation="silu",
        dtype=None,
        **kwargs,
    ):
        inputs = keras.Input(shape=(None, None, None, 3), dtype=dtype, name="images")

        # Patch embedding (3D Convolution)
        patch_embed = layers.Conv3D(
            filters=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            padding="valid",
            name="patch_embed",
        )
        x = patch_embed(inputs)

        # Transformer Blocks
        for i in range(depth):
            block = Qwen2VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                name=f"blocks.{i}",
            )
            x = block(x)

        # Output merger
        merger = layers.Conv2D(
            filters=hidden_size,
            kernel_size=2,
            strides=2,
            padding="valid",
            name="merger",
        )
        outputs = merger(x)

        super().__init__(inputs=inputs, outputs=outputs, dtype=dtype, **kwargs)

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "hidden_size": self.hidden_size,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "activation": self.activation,
            }
        )
        return config


class Qwen2VLVisionBlock(layers.Layer):
    """Single Transformer Block for Qwen2-VL Vision.

    Args:
        hidden_size: int. The embedding dimension.
        num_heads: int. Number of attention heads.
        mlp_ratio: int. Expansion ratio for the MLP.
        activation: str. Activation function.
        **kwargs: Standard Keras keyword arguments.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio, activation, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.activation = activation

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size // num_heads
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = layers.Sequential(
            [
                layers.Dense(int(hidden_size * mlp_ratio)),
                layers.Activation(activation),
                layers.Dense(hidden_size),
            ]
        )

    def call(self, x, grid_thw=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "activation": self.activation,
            }
        )
        return config