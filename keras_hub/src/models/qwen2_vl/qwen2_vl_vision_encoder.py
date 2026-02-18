import numpy as np
from keras import layers
from keras import ops


def _quick_gelu(x):
    """Quick GELU: x * sigmoid(1.702 * x)."""
    return x * ops.sigmoid(ops.cast(1.702, x.dtype) * x)


def _rotate_half(x):
    """Rotate last dim by splitting and negating halves."""
    half = ops.shape(x)[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return ops.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Apply RoPE to vision q and k.

    Args:
        q: (seq_len, num_heads, head_dim)
        k: (seq_len, num_heads, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    cos = ops.expand_dims(cos, axis=-2)
    sin = ops.expand_dims(sin, axis=-2)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2VLVisionRotaryEmbedding(layers.Layer):
    """Vision RoPE: returns raw frequency table of shape (seqlen, dim//2).

    Args:
        dim: int. Half of the attention head dimension (head_dim // 2).
        theta: float. RoPE base. Defaults to 10000.0.
    """

    def __init__(self, dim, theta=10000.0, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.theta = theta

        inv_freq_init = 1.0 / (
            theta ** (np.arange(0, dim, 2, dtype="float32") / dim)
        )
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(len(inv_freq_init),),
            initializer="zeros",
            trainable=False,
        )
        self.inv_freq.assign(inv_freq_init)

    def call(self, seqlen):
        seq = ops.cast(ops.arange(seqlen), "float32")
        freqs = ops.outer(seq, ops.cast(self.inv_freq, "float32"))
        return freqs

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "theta": self.theta})
        return config


class Qwen2VLVisionAttention(layers.Layer):
    """Fused-QKV self-attention with RoPE for the vision encoder.

    Has single qkv Dense (bias=True),
    proj Dense (bias=True), manual scaled dot-product attention.

    Args:
        embed_dim: int. Vision encoder embedding dimension.
        num_heads: int. Number of attention heads.
    """

    def __init__(self, embed_dim, num_heads, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = layers.Dense(
            embed_dim * 3, use_bias=True, dtype=dtype, name="qkv"
        )
        self.proj = layers.Dense(
            embed_dim, use_bias=True, dtype=dtype, name="proj"
        )

    def build(self, input_shape):
        self.qkv.build(input_shape)
        self.proj.build([None, self.embed_dim])
        self.built = True

    def call(self, x, position_embeddings=None):
        seq_len = ops.shape(x)[0]
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = ops.transpose(q, (1, 0, 2))
        k = ops.transpose(k, (1, 0, 2))
        v = ops.transpose(v, (1, 0, 2))
        attn = ops.matmul(q, ops.transpose(k, (0, 2, 1))) * self.scale
        attn = ops.softmax(ops.cast(attn, "float32"), axis=-1)
        attn = ops.cast(attn, q.dtype)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (1, 0, 2))
        out = ops.reshape(out, (seq_len, self.embed_dim))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"embed_dim": self.embed_dim, "num_heads": self.num_heads}
        )
        return config


class Qwen2VLVisionMlp(layers.Layer):
    """Two-layer MLP with quick_gelu for the vision transformer block.

    Args:
        embed_dim: int. Input/output dimension.
        mlp_dim: int. Hidden dimension.
    """

    def __init__(self, embed_dim, mlp_dim, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.fc1 = layers.Dense(mlp_dim, use_bias=True, dtype=dtype, name="fc1")
        self.fc2 = layers.Dense(
            embed_dim, use_bias=True, dtype=dtype, name="fc2"
        )

    def build(self, input_shape):
        self.fc1.build(input_shape)
        self.fc2.build([None, self.mlp_dim])
        self.built = True

    def call(self, x):
        return self.fc2(_quick_gelu(self.fc1(x)))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


class Qwen2VLVisionBlock(layers.Layer):
    """Single vision transformer block.

    Pre-norm, fused-QKV attention, quick_gelu MLP.

    Args:
        embed_dim: int. Vision encoder embedding dimension.
        num_heads: int. Number of attention heads.
        mlp_ratio: float. MLP hidden dim multiplier. Defaults to 4.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name="norm1"
        )
        self.attn = Qwen2VLVisionAttention(
            embed_dim=embed_dim, num_heads=num_heads, dtype=dtype, name="attn"
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name="norm2"
        )
        self.mlp = Qwen2VLVisionMlp(
            embed_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dtype=dtype,
            name="mlp",
        )

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.attn.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        self.built = True

    def call(self, x, position_embeddings=None):
        x = x + self.attn(
            self.norm1(x), position_embeddings=position_embeddings
        )
        x = x + self.mlp(self.norm2(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
            }
        )
        return config


class Qwen2VLPatchMerger(layers.Layer):
    """Merges spatial patches and projects to the LLM hidden dimension.

    Consists of:
    - LayerNorm on vision features.
    - Reshape: group spatial_merge_size² adjacent tokens.
    - Two-layer MLP (Linear → GELU → Linear).

    Args:
        hidden_size: int. Output dimension (LLM hidden size).
        embed_dim: int. Vision encoder embedding dimension.
        spatial_merge_size: int. Spatial merge factor. Defaults to 2.
    """

    def __init__(
        self, hidden_size, embed_dim, spatial_merge_size=2, dtype=None, **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.spatial_merge_size = spatial_merge_size
        self.mlp_hidden = embed_dim * (spatial_merge_size**2)
        self.ln_q = layers.LayerNormalization(
            epsilon=1e-6, dtype=dtype, name="ln_q"
        )
        self.mlp_fc1 = layers.Dense(
            self.mlp_hidden, use_bias=True, dtype=dtype, name="mlp_fc1"
        )
        self.mlp_fc2 = layers.Dense(
            hidden_size, use_bias=True, dtype=dtype, name="mlp_fc2"
        )

    def build(self, input_shape):
        self.ln_q.build(input_shape)
        self.mlp_fc1.build([None, self.mlp_hidden])
        self.mlp_fc2.build([None, self.mlp_hidden])
        self.built = True

    def call(self, x):
        x = self.ln_q(x)
        total = ops.shape(x)[0]
        merge_sq = self.spatial_merge_size**2
        x = ops.reshape(x, (total // merge_sq, self.mlp_hidden))
        x = ops.gelu(self.mlp_fc1(x))
        x = self.mlp_fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "embed_dim": self.embed_dim,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config


class Qwen2VLVisionEncoder(layers.Layer):
    """Qwen2-VL Vision Encoder (3D ViT with RoPE and PatchMerger).

    Accepts a flat patch tensor produced by ``Qwen2VLImageConverter``
    of shape ``(total_patches, C * temp_patch * patch²)``
    and a ``grid_thw`` tensor of shape ``(num_images, 3)``.

    Returns merged vision features of shape
    ``(total_patches // spatial_merge_size², hidden_size)``.

    Args:
        patch_size: int. Spatial patch size. Defaults to 14.
        temporal_patch_size: int. Temporal patch size. Defaults to 2.
        in_channels: int. Input image channels. Defaults to 3.
        embed_dim: int. ViT internal embedding dimension. Defaults to 1280.
        hidden_size: int. LLM hidden dimension (PatchMerger output).
            Defaults to 3584.
        depth: int. Number of vision transformer blocks. Defaults to 32.
        num_heads: int. Number of attention heads. Defaults to 16.
        mlp_ratio: float. MLP hidden dim multiplier. Defaults to 4.
        spatial_merge_size: int. Spatial merge factor. Defaults to 2.
    """

    def __init__(
        self,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=1280,
        hidden_size=3584,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        spatial_merge_size=2,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.spatial_merge_size = spatial_merge_size

        self.patch_embed = layers.Conv3D(
            filters=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            padding="valid",
            use_bias=False,
            dtype=dtype,
            name="patch_embed",
        )
        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = Qwen2VLVisionRotaryEmbedding(
            dim=head_dim // 2, theta=10000.0, dtype=dtype, name="rotary_pos_emb"
        )
        self.blocks = [
            Qwen2VLVisionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dtype=dtype,
                name=f"blocks_{i}",
            )
            for i in range(depth)
        ]
        self.merger = Qwen2VLPatchMerger(
            hidden_size=hidden_size,
            embed_dim=embed_dim,
            spatial_merge_size=spatial_merge_size,
            dtype=dtype,
            name="merger",
        )

    def build(self, input_shape=None):
        """Build all sublayers so their variables exist for weight loading."""
        # Conv3D sees 5-D input after reshape+transpose in call().
        conv_shape = (
            None,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        self.patch_embed.build(conv_shape)

        # Blocks and merger operate on (seq_len, embed_dim).
        block_shape = (None, self.embed_dim)
        for block in self.blocks:
            block.build(block_shape)
        self.merger.build(block_shape)
        self.built = True

    def _rot_pos_emb(self, grid_thw):
        """Build per-token (cos, sin) from grid_thw.

        Replicates HF rot_pos_emb: spatial-merge interleaved h/w pos ids,
        indexed into the rotary frequency table.

        Args:
            grid_thw: int tensor of shape (num_images, 3) — [t, h, w].

        Returns:
            Tuple (cos, sin) each of shape (total_tokens, head_dim).
        """
        pos_ids_list = []
        # grid_thw is a NumPy array at preprocessing time; use len() so this
        # works both eagerly and in traced graphs where shape[0] is known.
        num_images = (
            grid_thw.shape[0]
            if hasattr(grid_thw, "shape")
            else ops.shape(grid_thw)[0]
        )
        for i in range(int(num_images)):
            t = grid_thw[i, 0]
            h = grid_thw[i, 1]
            w = grid_thw[i, 2]
            hpos = ops.reshape(ops.arange(h), (h, 1))
            hpos = ops.broadcast_to(hpos, (h, w))
            hpos = ops.reshape(
                hpos,
                (
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ),
            )
            hpos = ops.transpose(hpos, (0, 2, 1, 3))
            hpos = ops.reshape(hpos, (-1,))
            wpos = ops.reshape(ops.arange(w), (1, w))
            wpos = ops.broadcast_to(wpos, (h, w))
            wpos = ops.reshape(
                wpos,
                (
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ),
            )
            wpos = ops.transpose(wpos, (0, 2, 1, 3))
            wpos = ops.reshape(wpos, (-1,))
            hw_ids = ops.stack([hpos, wpos], axis=-1)
            hw_ids = ops.tile(hw_ids, [int(t), 1])
            pos_ids_list.append(hw_ids)
        pos_ids = ops.concatenate(pos_ids_list, axis=0)
        max_grid_size = ops.max(grid_thw[:, 1:])
        rotary_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_emb = ops.take(
            rotary_emb_full, ops.reshape(pos_ids, (-1,)), axis=0
        )
        rotary_emb = ops.reshape(rotary_emb, (ops.shape(pos_ids)[0], -1))
        emb = ops.concatenate([rotary_emb, rotary_emb], axis=-1)
        cos = ops.cos(emb)
        sin = ops.sin(emb)
        return cos, sin

    def call(self, hidden_states, grid_thw=None):
        """Forward pass.

        Args:
            hidden_states: Flat patch tensor of shape
                ``(total_patches, C * temp_patch * patch²)``.
                Each row is one flattened patch group as produced by
                ``Qwen2VLImageConverter``.
            grid_thw: Int tensor of shape ``(num_images, 3)``.

        Returns:
            Merged features of shape
            ``(total_patches // spatial_merge_size², hidden_size)``.
        """
        # Reshape flat patches into 5D for Conv3D:
        # (N, in_channels, temporal_patch_size, patch_size, patch_size)
        hidden_states = ops.reshape(
            hidden_states,
            (
                -1,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ),
        )
        # Conv3D expects (batch, d, h, w, channels) in channels-last Keras.
        # Transpose from (N, C, T, P, P) → (N, T, P, P, C).
        hidden_states = ops.transpose(hidden_states, (0, 2, 3, 4, 1))
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = ops.reshape(hidden_states, (-1, self.embed_dim))

        position_embeddings = None
        if grid_thw is not None:
            cos, sin = self._rot_pos_emb(grid_thw)
            position_embeddings = (cos, sin)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, position_embeddings=position_embeddings
            )

        hidden_states = self.merger(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "embed_dim": self.embed_dim,
                "hidden_size": self.hidden_size,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "spatial_merge_size": self.spatial_merge_size,
            }
        )
        return config
