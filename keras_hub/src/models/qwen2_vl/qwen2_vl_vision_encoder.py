import keras
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_hub.src.models.backbone import Backbone


class Qwen2VLVisionEncoder(Backbone):
    def __init__(
        self,
        patch_size=14,
        temporal_patch_size=2,
        hidden_size=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=4,
        activation="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.activation = activation

        # 3D convolution to handle both Video (Time) and Images
        self.patch_embed = layers.Conv3D(
            filters=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            padding="valid",
            name="patch_embed",
        )

        # Placeholder for Qwen2VL transformer blocks
        self.blocks = [
            Qwen2VLVisionBlock(hidden_size, num_heads, mlp_ratio, activation, name=f"blocks.{i}")
            for i in range(depth)
        ]

        # Patch Merger to downsample tokens before sending to LLM
        self.merger = layers.Conv2D(
            filters=hidden_size,
            kernel_size=2,
            strides=2,
            padding="valid",
            name="merger",
        )

    def call(self, x, grid_thw=None):
        # x shape: (Batch, Time, Height, Width, Channels)
        x = self.patch_embed(x) 
        
        # Note: 3D-RoPE implementation pending
        
        for block in self.blocks:
            x = block(x, grid_thw=grid_thw)

        x = self.merger(x)
        
        return x

class Qwen2VLVisionBlock(layers.Layer):
    def __init__(self, hidden_size, num_heads, mlp_ratio, activation, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size//num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = layers.Dense(hidden_size * mlp_ratio)

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