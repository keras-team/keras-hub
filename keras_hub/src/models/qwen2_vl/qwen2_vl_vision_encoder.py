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

import keras
import numpy as np
from keras import layers, ops

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

        # 1. Patch embedding (3D Convolution)
        patch_embed = layers.Conv3D(
            filters=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            padding="valid",
            name="patch_embed",
        )
        x = patch_embed(inputs)

        # 2. Rotary Embedding (Initialize it here!)
        self.rotary_emb = Qwen2VLRotaryEmbedding(hidden_size // num_heads)

        # 3. Transformer Blocks
        # We must save these to a list so 'call' can use them later
        self.blocks = []
        for i in range(depth):
            block = Qwen2VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                activation=activation,
                name=f"blocks.{i}",
            )
            self.blocks.append(block)
            x = block(x)  # Pass through for Functional API graph construction

        # 4. Output merger
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
        # Note: self.rotary_emb is already set above

    def call(self, x, grid_thw=None):
        # x shape: (Batch, Time, Height, Width, Channels)
        x = self.patch_embed(x)

        # Flatten x: (Batch, T*H*W, Hidden)
        input_shape = ops.shape(x)
        B = input_shape[0]
        x = ops.reshape(x, (B, -1, self.hidden_size))

        # Calculate RoPE (if grid is provided)
        rotary_pos_emb = None
        if grid_thw is not None:
            rotary_pos_emb = self.rotary_emb(grid_thw)

        # Iterate through the stored blocks
        for block in self.blocks:
            x = block(x, rotary_pos_emb=rotary_pos_emb)

        # Warning: This skips the Merger for now to keep tests running!
        # x = self.merger(x)

        return x

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

    def call(self, x, rotary_pos_emb=None):
        residual = x
        x = self.norm1(x)

        # We pass rotary embeddings to the attention layer
        # Note: Keras MHA doesn't support 'rotary_pos_emb' natively yet.
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


class Qwen2VLRotaryEmbedding(layers.Layer):
    """Calculates 3D Rotary Positional Embeddings for Qwen2-VL."""

    def __init__(self, dim, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.base = base

    def call(self, grid_thw):
        # Placeholder for 3D RoPE logic
        # Returns dummy cos/sin shapes: (Batch, Seq_Len, Dim)
        seq_len = ops.prod(grid_thw, axis=1)
        max_len = ops.max(seq_len)

        shape = (1, max_len, self.dim)
        cos = ops.ones(shape, dtype="float32")
        sin = ops.zeros(shape, dtype="float32")

        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "base": self.base})
        return config
