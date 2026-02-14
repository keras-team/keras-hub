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
import numpy as np
from keras import layers
from keras import ops

from keras_hub.src.models.backbone import Backbone


class Qwen2VLVisionEncoder(Backbone):
    """Qwen2-VL Vision Encoder (ViT).

    A 3D Vision Transformer backbone that processes video/image inputs
    using 3D convolution patch embeddings and rotary position embeddings.
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
        inputs = keras.Input(
            shape=(None, None, None, 3), dtype=dtype, name="images"
        )

        # 1. Patch Embedding (3D Convolution)
        self.patch_embed = layers.Conv3D(
            filters=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            padding="valid",
            name="patch_embed",
        )
        x = self.patch_embed(inputs)

        # Flatten spatial dims: (Batch, Seq_Len, Hidden)
        # Keeps Batch dim dynamic, calculates Seq_Len, keeps Hidden fixed.
        x = ops.reshape(x, (ops.shape(x)[0], -1, hidden_size))

        # 2. Rotary Embedding
        self.rotary_emb = Qwen2VLRotaryEmbedding(hidden_size // num_heads)

        # 3. Transformer Blocks
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
            x = block(x)

        outputs = x
        super().__init__(inputs=inputs, outputs=outputs, dtype=dtype, **kwargs)

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.activation = activation

    def call(self, x, grid_thw=None):
        x = self.patch_embed(x)

        # Capture dynamic shapes for restoration later
        shape = ops.shape(x)
        B, T, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]

        # Flatten for Transformer
        x = ops.reshape(x, (B, -1, self.hidden_size))

        # Calculate RoPE if grid info is provided
        rotary_pos_emb = None
        if grid_thw is not None:
            rotary_pos_emb = self.rotary_emb(grid_thw)

        for block in self.blocks:
            x = block(x, rotary_pos_emb=rotary_pos_emb)

        # Restore 5D shape: (Batch, Time, Height, Width, Channels)
        x = ops.reshape(x, (B, T, H, W, C))

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
    """Single Transformer Block for Qwen2-VL Vision."""

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

        self.mlp = keras.Sequential(
            [
                layers.Dense(int(hidden_size * mlp_ratio)),
                layers.Activation(activation),
                layers.Dense(hidden_size),
            ]
        )

    def call(self, x, rotary_pos_emb=None):
        residual = x
        x = self.norm1(x)
        # Note: Pass rotary embeddings here when Keras MHA supports it fully,
        # or implement custom attention if needed. For now, standard MHA.
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
    """Calculates 3D Rotary Positional Embeddings."""

    def __init__(self, dim, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.base = base
        self.inv_freq = self._compute_inv_freq(dim, base)

    def _compute_inv_freq(self, dim, base):
        exponent = np.arange(0, dim, 2).astype("float32")
        value = exponent / dim
        inv_freq = 1.0 / (base**value)
        return inv_freq

    def call(self, grid_thw):
        # Implementation of 3D RoPE (Time, Height, Width)
        max_t = ops.max(grid_thw[:, 0])
        max_h = ops.max(grid_thw[:, 1])
        max_w = ops.max(grid_thw[:, 2])

        t_pos = ops.arange(max_t, dtype="float32")
        h_pos = ops.arange(max_h, dtype="float32")
        w_pos = ops.arange(max_w, dtype="float32")

        inv_freq_tensor = ops.convert_to_tensor(self.inv_freq, dtype="float32")

        t_emb = ops.outer(t_pos, inv_freq_tensor)
        h_emb = ops.outer(h_pos, inv_freq_tensor)
        w_emb = ops.outer(w_pos, inv_freq_tensor)

        t_emb = ops.concatenate([t_emb, t_emb], axis=-1)
        h_emb = ops.concatenate([h_emb, h_emb], axis=-1)
        w_emb = ops.concatenate([w_emb, w_emb], axis=-1)

        return t_emb, h_emb, w_emb

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "base": self.base})
        return config
