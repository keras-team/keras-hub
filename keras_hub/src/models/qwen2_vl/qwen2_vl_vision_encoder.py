import math

import keras
from keras import ops


class Qwen2VLVisionAttention(keras.layers.Layer):
    """Multi-head self-attention for vision tokens with 2D RoPE.

    Attention is computed independently per (image, temporal frame) segment
    using ``cu_seqlens``, matching the HF implementation where each temporal
    patch of each image attends only within itself.

    Args:
        embed_dim: int. Input embedding dimension.
        num_heads: int. Number of attention heads.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        # Fused QKV projection, matching HF `attn.qkv`.
        self.qkv = keras.layers.Dense(
            self.embed_dim * 3,
            use_bias=True,
            dtype=self.dtype_policy,
            name="qkv",
        )
        self.qkv.build(input_shape)

        self.proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="proj",
        )
        self.proj.build((input_shape[0], self.embed_dim))
        self.built = True

    def call(self, x, position_embeddings, segment_ids=None):
        """
        Args:
            x: Tensor of shape ``(seq_len, embed_dim)``.
            position_embeddings: tuple of ``(cos, sin)``, each of shape
                ``(seq_len, head_dim)``.
            segment_ids: int32 tensor ``(seq_len,)`` assigning each patch to
                an (image, frame) segment. Patches only attend to other
                patches in the same segment (HF ``cu_seqlens`` semantics,
                expressed as a mask so this stays traceable under
                ``tf.function``). ``None`` means full attention over ``x``.
        Returns:
            Tensor of shape ``(seq_len, embed_dim)``.
        """
        seq_len = ops.shape(x)[0]

        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (S, H, D)

        cos_emb, sin_emb = position_embeddings
        cos_emb = ops.expand_dims(cos_emb, axis=1)  # (S, 1, D)
        sin_emb = ops.expand_dims(sin_emb, axis=1)

        def _apply_rope(t):
            t_dtype = t.dtype
            t = ops.cast(t, "float32")
            half = self.head_dim // 2
            t1, t2 = t[..., :half], t[..., half:]
            rotated = ops.concatenate([-t2, t1], axis=-1)
            return ops.cast(t * cos_emb + rotated * sin_emb, t_dtype)

        q = _apply_rope(q)
        k = _apply_rope(k)

        q = ops.transpose(q, (1, 0, 2))  # (H, S, D)
        k = ops.transpose(k, (1, 0, 2))
        v = ops.transpose(v, (1, 0, 2))

        # Per-frame attention: mask out cross-segment attention. For a
        # single segment the mask is all-true (no-op).
        scores = ops.einsum("hid,hjd->hij", q, k) * self.scale
        scores = ops.cast(scores, "float32")
        if segment_ids is not None:
            same = ops.equal(
                ops.expand_dims(segment_ids, axis=1),
                ops.expand_dims(segment_ids, axis=0),
            )
            scores = ops.where(
                ops.expand_dims(same, axis=0), scores, float("-inf")
            )
        scores = ops.softmax(scores, axis=-1)
        scores = ops.cast(scores, self.compute_dtype)
        out = ops.einsum("hij,hjd->hid", scores, v)

        out = ops.transpose(out, (1, 0, 2))  # (S, H, D)
        out = ops.reshape(out, (seq_len, self.embed_dim))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"embed_dim": self.embed_dim, "num_heads": self.num_heads}
        )
        return config


class Qwen2VLVisionMLP(keras.layers.Layer):
    """Vision MLP: fc1 -> GELU -> fc2.

    HF uses ``QuickGELU`` (i.e. ``x * sigmoid(1.702x)``) in vision blocks.
    """

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.fc1 = keras.layers.Dense(
            self.mlp_dim, use_bias=True, dtype=self.dtype_policy, name="fc1"
        )
        self.fc1.build(input_shape)
        self.fc2 = keras.layers.Dense(
            self.embed_dim, use_bias=True, dtype=self.dtype_policy, name="fc2"
        )
        self.fc2.build((input_shape[0], self.mlp_dim))
        self.built = True

    def call(self, x):
        x = self.fc1(x)
        # QuickGELU: x * sigmoid(1.702 * x)
        x = x * ops.sigmoid(x * 1.702)
        return self.fc2(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


class Qwen2VLVisionBlock(keras.layers.Layer):
    """Single vision transformer block: LN -> Attn -> LN -> MLP.

    Uses standard ``LayerNormalization`` (with bias) and QuickGELU MLP.
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.epsilon = epsilon

    def build(self, input_shape):
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype="float32", name="norm1"
        )
        self.norm1.build(input_shape)
        self.attn = Qwen2VLVisionAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.attn.build(input_shape)
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype="float32", name="norm2"
        )
        self.norm2.build(input_shape)
        self.mlp = Qwen2VLVisionMLP(
            embed_dim=self.embed_dim,
            mlp_dim=self.mlp_dim,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(input_shape)
        self.built = True

    def call(self, x, position_embeddings, segment_ids=None):
        # Vision always runs in float32 for stability.
        x = ops.cast(x, "float32")
        normed = self.norm1(x)
        normed = ops.cast(normed, self.compute_dtype)
        attn_out = self.attn(normed, position_embeddings, segment_ids)
        x = x + ops.cast(attn_out, "float32")

        normed = self.norm2(x)
        normed = ops.cast(normed, self.compute_dtype)
        mlp_out = self.mlp(normed)
        return x + ops.cast(mlp_out, "float32")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "epsilon": self.epsilon,
            }
        )
        return config


class Qwen2VLPatchMerger(keras.layers.Layer):
    """Patch merger: LayerNorm -> MLP to project merged spatial groups.

    Groups ``spatial_merge_size²`` patches into one token: LN on each patch,
    concat the group, fc1 -> exact GELU -> fc2.

    Args:
        embed_dim: int. Vision embed dimension.
        out_dim: int. Output dimension (= text backbone hidden dim).
        spatial_merge_size: int. Merge factor along each spatial dim.
        epsilon: float. LayerNorm epsilon.
    """

    def __init__(
        self, embed_dim, out_dim, spatial_merge_size, epsilon, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.spatial_merge_size = spatial_merge_size
        self.epsilon = epsilon
        self.merged_dim = embed_dim * spatial_merge_size * spatial_merge_size

    def build(self, input_shape):
        self.ln_q = keras.layers.LayerNormalization(
            epsilon=self.epsilon, dtype="float32", name="ln_q"
        )
        self.ln_q.build((None, self.embed_dim))
        self.mlp_fc1 = keras.layers.Dense(
            self.merged_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="mlp_fc1",
        )
        self.mlp_fc1.build((None, self.merged_dim))
        self.mlp_fc2 = keras.layers.Dense(
            self.out_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="mlp_fc2",
        )
        self.mlp_fc2.build((None, self.merged_dim))
        self.built = True

    def call(self, x):
        """
        Args:
            x: Tensor ``(total_patches, embed_dim)``.
        Returns:
            Tensor ``(total_patches // merge_size², out_dim)``.
        """
        x = ops.cast(x, "float32")
        x = self.ln_q(x)
        num_patches = ops.shape(x)[0]
        ms2 = self.spatial_merge_size * self.spatial_merge_size
        x = ops.reshape(x, (num_patches // ms2, ms2 * self.embed_dim))
        x = ops.cast(x, self.compute_dtype)
        x = self.mlp_fc1(x)
        # HF uses nn.GELU() — exact, not the tanh approximation.
        x = ops.gelu(x, approximate=False)
        return self.mlp_fc2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "out_dim": self.out_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "epsilon": self.epsilon,
            }
        )
        return config


class Qwen2VLVisionEncoder(keras.Model):
    """Vision encoder for Qwen2-VL (ViT with spatial patch merger).

    Processes image patches of shape ``(N, T, pH, pW, C)`` through:

    1. Conv3D patch embedding
    2. Per-frame ViT blocks with 2D axial RoPE
    3. Spatial patch merger → text hidden dimension

    Args:
        patch_size: int. Spatial patch size in pixels.
        temporal_patch_size: int. Temporal patch size.
        in_channels: int. Input channels (typically 3).
        embed_dim: int. Vision hidden dimension.
        out_dim: int. Output dimension (= text backbone ``hidden_dim``).
        depth: int. Number of ViT blocks.
        num_heads: int. Attention heads per ViT block.
        mlp_ratio: float. MLP hidden dim = ``embed_dim * mlp_ratio``.
        spatial_merge_size: int. Spatial merge factor.
        theta: float. RoPE base frequency.
        epsilon: float. LayerNorm epsilon.
        dtype: compute dtype; the vision tower always runs in float32.
    """

    def __init__(
        self,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        embed_dim=1280,
        out_dim=None,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        spatial_merge_size=2,
        theta=10000.0,
        epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        # Always run the vision encoder in float32 for numerical stability.
        if dtype is not None and dtype != "float32":
            dtype = "float32"
        super().__init__(dtype=dtype, **kwargs)

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_dim = out_dim if out_dim is not None else embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.spatial_merge_size = spatial_merge_size
        self.theta = theta
        self.epsilon = epsilon

        self.head_dim = embed_dim // num_heads
        mlp_dim = int(embed_dim * mlp_ratio)

        # Vision rotary inverse frequencies (matches HF `persistent=False`,
        # i.e. computed at runtime, not a stored weight).
        spatial_dim = self.head_dim // 2
        self._inv_freq = [
            1.0 / (math.pow(theta, i / spatial_dim))
            for i in range(0, spatial_dim, 2)
        ]

        # Conv3D patch embedding (no bias, matching HF `visual.patch_embed`).
        self.patch_embed = keras.layers.Conv3D(
            filters=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            padding="valid",
            use_bias=False,
            dtype=self.dtype_policy,
            name="patch_embed",
        )

        self.blocks = [
            Qwen2VLVisionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                epsilon=epsilon,
                dtype=self.dtype_policy,
                name=f"blocks_{i}",
            )
            for i in range(depth)
        ]

        self.merger = Qwen2VLPatchMerger(
            embed_dim=embed_dim,
            out_dim=self.out_dim,
            spatial_merge_size=spatial_merge_size,
            epsilon=epsilon,
            dtype=self.dtype_policy,
            name="merger",
        )

    def build(self, input_shape=None):
        """Build all sublayers so weights exist before any forward pass."""
        if not self.patch_embed.built:
            self.patch_embed.build(
                (
                    None,
                    self.temporal_patch_size,
                    self.patch_size,
                    self.patch_size,
                    self.in_channels,
                )
            )
        for blk in self.blocks:
            if not blk.built:
                blk.build((None, self.embed_dim))
        if not self.merger.built:
            self.merger.build((None, self.embed_dim))
        super().build(input_shape)

    def _rot_pos_emb(self, grid_thw):
        """Compute 2D axial RoPE (cos, sin) for all image patches.

        Produces block-major (merge-block-ordered) h/w indices per image,
        repeated for each temporal frame — matching HF
        ``get_vision_position_ids``.

        Implemented with tensor ops only so it traces under ``tf.function``
        (generate compiles ``generate_step``).

        Args:
            grid_thw: int tensor ``(num_images, 3)`` with [T, H, W] per image.
        Returns:
            (cos_emb, sin_emb): each ``(total_tokens, head_dim)``.
        """
        grid_thw = ops.cast(grid_thw, "int32")
        ms = self.spatial_merge_size
        n = grid_thw.shape[0]
        max_hw = ops.max(grid_thw[:, 1:])
        inv_freq = ops.cast(ops.array(self._inv_freq), "float32")
        positions = ops.cast(ops.arange(max_hw), "float32")
        freq_table = ops.einsum("i,j->ij", positions, inv_freq)  # (hw, hd//4)

        all_embeds = []
        for idx in range(n):
            t_val = grid_thw[idx][0]
            h_val = grid_thw[idx][1]
            w_val = grid_thw[idx][2]
            merged_h = h_val // ms
            merged_w = w_val // ms

            # Block-major merge ordering: (merged_h, merged_w, ms, ms).
            row_idx = ops.expand_dims(
                ops.arange(merged_h), axis=(1, 2, 3)
            ) * ms + ops.expand_dims(
                ops.arange(ms), axis=(0, 1, 3)
            )  # (merged_h, 1, ms, 1)
            col_idx = ops.expand_dims(
                ops.arange(merged_w), axis=(0, 2, 3)
            ) * ms + ops.expand_dims(
                ops.arange(ms), axis=(0, 1, 2)
            )  # (1, merged_w, 1, ms)
            row_idx = ops.broadcast_to(row_idx, (merged_h, merged_w, ms, ms))
            col_idx = ops.broadcast_to(col_idx, (merged_h, merged_w, ms, ms))
            row_idx = ops.reshape(row_idx, (-1,))
            col_idx = ops.reshape(col_idx, (-1,))

            row_freqs = ops.take(freq_table, row_idx, axis=0)  # (h*w, hd//4)
            col_freqs = ops.take(freq_table, col_idx, axis=0)
            spatial_emb = ops.concatenate(
                [row_freqs, col_freqs], axis=-1
            )  # (h*w, head_dim//2)

            # Tile the spatial embedding across all temporal frames.
            n_hw = ops.shape(spatial_emb)[0]
            h_dim = ops.shape(spatial_emb)[1]
            spatial_emb = ops.broadcast_to(
                ops.expand_dims(spatial_emb, axis=0), (t_val, n_hw, h_dim)
            )
            spatial_emb = ops.reshape(spatial_emb, (-1, h_dim))

            all_embeds.append(spatial_emb)

        rotary = ops.concatenate(all_embeds, axis=0)  # (N, head_dim // 2)
        # Duplicate to full head_dim, matching HF `cat([emb, emb], -1)`.
        rotary = ops.concatenate([rotary, rotary], axis=-1)
        return ops.cos(rotary), ops.sin(rotary)

    def _segment_ids(self, grid_thw):
        """Per-patch (image, frame) segment ids for attention.

        Each of the ``t`` frames of an image is its own attention segment
        (HF ``cu_seqlens`` semantics). Returns a ``(total_patches,)`` int32
        tensor; the loop bound ``num_images`` is a static shape so this
        stays traceable under ``tf.function``.
        """
        grid_thw = ops.cast(grid_thw, "int32")
        t_counts = grid_thw[:, 0]
        hw = grid_thw[:, 1] * grid_thw[:, 2]
        # Segment id of frame 0 for each image.
        seg_offsets = ops.cumsum(t_counts) - t_counts
        segments = []
        for i in range(grid_thw.shape[0]):
            t_i = t_counts[i]
            hw_i = hw[i]
            # Patch index within the image → frame index.
            frame_of_patch = ops.arange(t_i * hw_i) // hw_i
            segments.append(frame_of_patch + seg_offsets[i])
        return ops.cast(ops.concatenate(segments, axis=0), "int32")

    def call(self, pixel_values, grid_thw):
        """Forward pass through the vision encoder.

        Args:
            pixel_values: patches ``(total_patches, T, pH, pW, C)`` or
                batched ``(batch, total_patches, T, pH, pW, C)``.
            grid_thw: int tensor ``(num_images, 3)`` or batched
                ``(batch, num_images, 3)`` with [T, H, W] per image.
        Returns:
            Unbatched: ``(total_merged_tokens, out_dim)``.
            Batched: ``(batch, total_merged_tokens, out_dim)``.
        """
        batched = len(ops.shape(pixel_values)) == 6
        if batched:
            # Collapse batch: (B, N, T, pH, pW, C) -> (B*N, T, pH, pW, C).
            pixel_values = ops.reshape(
                pixel_values,
                (
                    -1,
                    self.temporal_patch_size,
                    self.patch_size,
                    self.patch_size,
                    self.in_channels,
                ),
            )
            grid_thw = ops.reshape(grid_thw, (-1, 3))

        # Early return for zero-sized input (text-only on multimodal model).
        if ops.shape(pixel_values)[0] == 0:
            empty = ops.zeros((0, self.out_dim), dtype=self.compute_dtype)
            if batched:
                empty = ops.expand_dims(empty, axis=0)
            return empty

        # Patch embedding: each patch is one Conv3D "sample".
        hidden_states = self.patch_embed(pixel_values)  # (N,1,1,1,E)
        hidden_states = ops.reshape(
            hidden_states, (ops.shape(hidden_states)[0], self.embed_dim)
        )
        hidden_states = ops.cast(hidden_states, "float32")

        position_embeddings = self._rot_pos_emb(grid_thw)
        segment_ids = self._segment_ids(grid_thw)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings, segment_ids)

        merged = self.merger(hidden_states)

        if batched:
            merged = ops.expand_dims(merged, axis=0)
        return merged

    def compute_output_spec(self, pixel_values, grid_thw=None):
        """Infer output spec for backbone functional-graph construction."""
        if len(pixel_values.shape) == 6:  # batched
            return keras.KerasTensor(
                shape=(pixel_values.shape[0], None, self.out_dim),
                dtype="float32",
            )
        return keras.KerasTensor(
            shape=(None, self.out_dim),
            dtype="float32",
        )

    def get_config(self):
        return {
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "out_dim": self.out_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "spatial_merge_size": self.spatial_merge_size,
            "theta": self.theta,
            "epsilon": self.epsilon,
        }
