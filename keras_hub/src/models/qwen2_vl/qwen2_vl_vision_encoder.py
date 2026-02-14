"""Qwen2-VL Vision Encoder.

This module implements the Vision Transformer (ViT) for Qwen2-VL,
including 3D patch embedding, 2D rotary position embeddings,
vision attention, transformer blocks, and spatial patch merging.
"""

import keras
from keras import ops

from keras_hub.src.models.qwen2_vl.qwen2_vl_attention import _rotate_half
from keras_hub.src.utils.keras_utils import clone_initializer


class Qwen2VLPatchEmbed(keras.layers.Layer):
    """3D convolution-based patch embedding for Qwen2-VL.

    Processes image/video frames using a 3D convolution with kernel size
    `(temporal_patch_size, patch_size, patch_size)` to produce patch embeddings.

    Args:
        patch_size: int. Spatial patch size (height and width).
        temporal_patch_size: int. Temporal patch size (number of frames
            grouped together). Defaults to `2`.
        in_channels: int. Number of input channels. Defaults to `3`.
        embed_dim: int. Embedding dimension. Defaults to `1152`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype
            for computations and weights.
    """

    def __init__(
        self,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=1152,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = keras.layers.Conv3D(
            filters=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            use_bias=False,
            data_format="channels_first",
            dtype=dtype,
            name="proj",
        )

    def call(self, hidden_states):
        """Processes input patches through the 3D convolution.

        Args:
            hidden_states: Tensor of shape
                `(total_patches, in_channels, temporal_patch_size,
                  patch_size, patch_size)`.

        Returns:
            Tensor of shape `(total_patches, embed_dim)`.
        """
        hidden_states = self.proj(hidden_states)
        # Flatten spatial and temporal dims: (batch, embed_dim, 1, 1, 1) -> (batch, embed_dim)
        hidden_states = ops.reshape(hidden_states, (-1, self.embed_dim))
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class Qwen2VLVisionRotaryEmbedding(keras.layers.Layer):
    """2D Rotary position embedding for the Qwen2-VL vision encoder.

    Computes rotary embeddings from spatial position indices. The embedding
    dimension is split in half: one half for height positions and the other
    half for width positions.

    Args:
        dim: int. Dimension of the rotary embedding (typically `head_dim // 2`).
        theta: float. Base frequency for the rotary embedding.
            Defaults to `10000.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype
            for computations.
    """

    def __init__(self, dim, theta=10000.0, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.theta = theta
        # Compute inverse frequencies: 1 / (theta ^ (2i / dim))
        inv_freq = 1.0 / (
            theta
            ** (
                ops.cast(ops.arange(0, dim, 2), "float32")
                / dim
            )
        )
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=inv_freq.shape,
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        self.inv_freq.assign(inv_freq)

    def call(self, seqlen):
        """Computes rotary embeddings for a given sequence length.

        Args:
            seqlen: int. The maximum sequence length (max grid dimension).

        Returns:
            Tensor of shape `(seqlen, dim // 2)` containing the rotary
            frequencies.
        """
        seq = ops.cast(ops.arange(seqlen), "float32")
        # Outer product: (seqlen,) x (dim//2,) -> (seqlen, dim//2)
        freqs = ops.einsum("i,j->ij", seq, self.inv_freq)
        return freqs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "theta": self.theta,
            }
        )
        return config


class Qwen2VLVisionMLP(keras.layers.Layer):
    """MLP block for the Qwen2-VL vision encoder.

    A two-layer feedforward network with GELU activation.

    Args:
        dim: int. Input/output dimension.
        hidden_dim: int. Hidden dimension.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype
            for computations and weights.
    """

    def __init__(self, dim, hidden_dim, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.fc1 = keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            dtype=dtype,
            name="fc1",
        )
        self.fc2 = keras.layers.Dense(
            dim,
            use_bias=True,
            dtype=dtype,
            name="fc2",
        )

    def call(self, x):
        x = self.fc1(x)
        x = ops.gelu(x, approximate=False)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Applies rotary position embedding to query and key tensors.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine part of rotary embedding.
        sin: Sine part of rotary embedding.

    Returns:
        Tuple of rotated query and key tensors.
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q = ops.cast(q, "float32")
    k = ops.cast(k, "float32")
    cos = ops.cast(ops.expand_dims(cos, axis=-2), "float32")
    sin = ops.cast(ops.expand_dims(sin, axis=-2), "float32")


    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return ops.cast(q_embed, orig_q_dtype), ops.cast(k_embed, orig_k_dtype)


class Qwen2VLVisionAttention(keras.layers.Layer):
    """Multi-head attention for the Qwen2-VL vision encoder.

    Uses a fused QKV projection and 2D rotary position embeddings.

    Args:
        embed_dim: int. Embedding dimension.
        num_heads: int. Number of attention heads.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype
            for computations and weights.
    """

    def __init__(self, embed_dim, num_heads, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = keras.layers.Dense(
            embed_dim * 3,
            use_bias=True,
            dtype=dtype,
            name="qkv",
        )
        self.proj = keras.layers.Dense(
            embed_dim,
            use_bias=True,
            dtype=dtype,
            name="proj",
        )

    def call(self, hidden_states, position_embeddings=None):
        """Forward pass of vision attention.

        Args:
            hidden_states: Tensor of shape `(seq_len, embed_dim)`.
                Note: no batch dim — all images are concatenated.
            position_embeddings: Tuple of `(cos, sin)` for rotary
                embeddings, each of shape `(seq_len, head_dim)`.

        Returns:
            Tensor of shape `(seq_len, embed_dim)`.
        """
        seq_length = ops.shape(hidden_states)[0]

        # QKV projection: (seq_len, 3 * embed_dim)
        qkv = self.qkv(hidden_states)
        # Reshape to (seq_len, 3, num_heads, head_dim)
        qkv = ops.reshape(
            qkv, (seq_length, 3, self.num_heads, self.head_dim)
        )
        # Transpose to (3, seq_len, num_heads, head_dim)
        qkv = ops.transpose(qkv, (1, 0, 2, 3))
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = _apply_rotary_pos_emb_vision(query, key, cos, sin)

        # Transpose for attention: (1, num_heads, seq_len, head_dim)
        query = ops.transpose(
            ops.expand_dims(query, axis=0), (0, 2, 1, 3)
        )
        key = ops.transpose(
            ops.expand_dims(key, axis=0), (0, 2, 1, 3)
        )
        value = ops.transpose(
            ops.expand_dims(value, axis=0), (0, 2, 1, 3)
        )

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        attn_weights = attn_weights * scale
        attn_weights = ops.softmax(
            ops.cast(attn_weights, "float32"), axis=-1
        )
        attn_weights = ops.cast(attn_weights, query.dtype)

        attn_output = ops.matmul(attn_weights, value)
        # (1, num_heads, seq_len, head_dim) -> (seq_len, embed_dim)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (seq_length, -1))

        attn_output = self.proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class Qwen2VLVisionBlock(keras.layers.Layer):
    """A single transformer block for the Qwen2-VL vision encoder.

    Pre-norm architecture: LN → Attention → residual → LN → MLP → residual.

    Args:
        embed_dim: int. Embedding dimension.
        num_heads: int. Number of attention heads.
        mlp_ratio: float. Ratio of MLP hidden dim to embed_dim.
            Defaults to `4.0`.
        layer_norm_epsilon: float. Epsilon for layer normalization.
            Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype
            for computations and weights.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.layer_norm_epsilon = layer_norm_epsilon

        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="norm1",
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="norm2",
        )
        self.attn = Qwen2VLVisionAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dtype=dtype,
            name="attn",
        )
        self.mlp = Qwen2VLVisionMLP(
            dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dtype=dtype,
            name="mlp",
        )

    def call(self, hidden_states, position_embeddings=None):
        """Forward pass through the vision block.

        Args:
            hidden_states: Tensor of shape `(seq_len, embed_dim)`.
            position_embeddings: Tuple of `(cos, sin)` for rotary
                embeddings.

        Returns:
            Tensor of shape `(seq_len, embed_dim)`.
        """
        # Self-attention with residual
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
        )
        # MLP with residual
        hidden_states = hidden_states + self.mlp(
            self.norm2(hidden_states),
        )
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class Qwen2VLPatchMerger(keras.layers.Layer):
    """Spatial patch merger for the Qwen2-VL vision encoder.

    Merges `spatial_merge_size × spatial_merge_size` adjacent patches into
    a single token, reducing the number of vision tokens by
    `spatial_merge_size²`.

    Architecture: LayerNorm → reshape (group patches) → Dense → GELU → Dense.

    Args:
        hidden_size: int. Output dimension (the LLM hidden dimension).
        context_dim: int. The ViT embedding dimension.
        spatial_merge_size: int. Size of the spatial merge window.
            Defaults to `2`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype
            for computations and weights.
    """

    def __init__(
        self,
        hidden_size,
        context_dim,
        spatial_merge_size=2,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size

        merge_hidden = context_dim * (spatial_merge_size ** 2)

        self.ln_q = keras.layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="ln_q",
        )
        self.dense1 = keras.layers.Dense(
            merge_hidden,
            use_bias=True,
            dtype=dtype,
            name="dense1",
        )
        self.dense2 = keras.layers.Dense(
            hidden_size,
            use_bias=True,
            dtype=dtype,
            name="dense2",
        )

    def call(self, x):
        """Merges adjacent patches.

        Args:
            x: Tensor of shape `(total_patches, context_dim)`.

        Returns:
            Tensor of shape `(merged_patches, hidden_size)`.
        """
        x = self.ln_q(x)
        merge_size = self.spatial_merge_size ** 2
        # Reshape to group adjacent patches
        x = ops.reshape(x, (-1, merge_size * self.context_dim))
        x = self.dense1(x)
        x = ops.gelu(x, approximate=False)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "context_dim": self.context_dim,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


class Qwen2VLVisionEncoder(keras.layers.Layer):
    """Qwen2-VL Vision Transformer encoder.

    Full vision encoder: PatchEmbed → 2D RoPE → VisionBlocks → PatchMerger.

    This encoder takes pixel values and grid dimensions as input, and
    produces vision embeddings suitable for interleaving with text token
    embeddings in the Qwen2-VL decoder.

    Args:
        hidden_size: int. The LLM hidden dimension (output dim of merger).
        embed_dim: int. ViT embedding dimension. Defaults to `1280`.
        depth: int. Number of vision transformer blocks. Defaults to `32`.
        num_heads: int. Number of attention heads. Defaults to `16`.
        patch_size: int. Spatial patch size. Defaults to `14`.
        temporal_patch_size: int. Temporal patch size. Defaults to `2`.
        in_channels: int. Number of input channels. Defaults to `3`.
        mlp_ratio: float. MLP hidden dim ratio. Defaults to `4.0`.
        spatial_merge_size: int. Spatial merge window size. Defaults to `2`.
        layer_norm_epsilon: float. Epsilon for layer norms.
            Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype
            for computations and weights.
    """

    def __init__(
        self,
        hidden_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        mlp_ratio=4.0,
        spatial_merge_size=2,
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.mlp_ratio = mlp_ratio
        self.spatial_merge_size = spatial_merge_size
        self.layer_norm_epsilon = layer_norm_epsilon

        head_dim = embed_dim // num_heads

        self.patch_embed = Qwen2VLPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dtype=dtype,
            name="patch_embed",
        )

        self.rotary_pos_emb = Qwen2VLVisionRotaryEmbedding(
            dim=head_dim // 2,
            dtype=dtype,
            name="rotary_pos_emb",
        )

        self.blocks = [
            Qwen2VLVisionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"block_{i}",
            )
            for i in range(depth)
        ]

        self.merger = Qwen2VLPatchMerger(
            hidden_size=hidden_size,
            context_dim=embed_dim,
            spatial_merge_size=spatial_merge_size,
            dtype=dtype,
            name="merger",
        )

    def _compute_rotary_pos_emb(self, grid_thw):
        """Computes 2D rotary position embeddings from grid dimensions.

        Args:
            grid_thw: Tensor of shape `(num_images, 3)` with
                `[temporal, height, width]` for each image/video.

        Returns:
            Tensor of shape `(total_patches, head_dim // 2)` with rotary
            frequencies.
        """
        all_pos_ids = []

        spatial_merge = self.spatial_merge_size
        for idx in range(ops.shape(grid_thw)[0]):
            t = grid_thw[idx, 0]
            h = grid_thw[idx, 1]
            w = grid_thw[idx, 2]

            # Height position IDs
            hpos_ids = ops.repeat(
                ops.expand_dims(ops.arange(h), axis=1), w, axis=1
            )
            # Reshape for spatial merge grouping
            hpos_ids = ops.reshape(
                hpos_ids,
                (h // spatial_merge, spatial_merge,
                 w // spatial_merge, spatial_merge),
            )
            hpos_ids = ops.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = ops.reshape(hpos_ids, (-1,))

            # Width position IDs
            wpos_ids = ops.repeat(
                ops.expand_dims(ops.arange(w), axis=0), h, axis=0
            )
            wpos_ids = ops.reshape(
                wpos_ids,
                (h // spatial_merge, spatial_merge,
                 w // spatial_merge, spatial_merge),
            )
            wpos_ids = ops.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = ops.reshape(wpos_ids, (-1,))

            # Stack [h, w] and repeat for each temporal frame
            pos_ids = ops.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids = ops.tile(pos_ids, (t, 1))
            all_pos_ids.append(pos_ids)

        pos_ids = ops.concatenate(all_pos_ids, axis=0)

        # Get max grid size for frequency computation
        max_grid_size = ops.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)

        # Gather embeddings for each position
        # pos_ids: (total_patches, 2) — height and width indices
        # rotary_pos_emb_full: (max_grid_size, dim//2)
        h_emb = ops.take(rotary_pos_emb_full, pos_ids[:, 0], axis=0)
        w_emb = ops.take(rotary_pos_emb_full, pos_ids[:, 1], axis=0)
        # Concatenate height and width embeddings along last dim
        rotary_pos_emb = ops.concatenate([h_emb, w_emb], axis=-1)

        return rotary_pos_emb

    def call(self, hidden_states, grid_thw):
        """Forward pass of the vision encoder.

        Args:
            hidden_states: Pixel values tensor of shape
                `(total_patches, in_channels, temporal_patch_size,
                  patch_size, patch_size)`.
            grid_thw: Tensor of shape `(num_images, 3)` containing
                `[temporal, height, width]` grid dimensions for each
                image/video.

        Returns:
            Tensor of shape `(merged_total_patches, hidden_size)`.
        """
        hidden_states = self.patch_embed(hidden_states)

        # Compute rotary position embeddings
        rotary_pos_emb = self._compute_rotary_pos_emb(grid_thw)
        emb = ops.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (
            ops.cos(ops.cast(emb, "float32")),
            ops.sin(ops.cast(emb, "float32")),
        )

        # Apply transformer blocks
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        # Spatial merge
        merged = self.merger(hidden_states)
        return merged

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "mlp_ratio": self.mlp_ratio,
                "spatial_merge_size": self.spatial_merge_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
