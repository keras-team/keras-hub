"""Qwen3.5 Vision Encoder.

Implements the vision transformer used by Qwen3.5-VL (images only for v1).
Architecture mirrors the HF Qwen3VLVisionModel:
  PatchEmbed(Conv3D) → pos_embed + rotary_pos_emb → ViT blocks → PatchMerger

Token count per image: grid_thw.prod() // spatial_merge_size^2
"""

import math

import keras
from keras import ops


class Qwen3_5VisionRotaryEmbedding(keras.layers.Layer):
    """1D rotary position embedding for the vision transformer (2D spatial).

    Produces cos/sin embeddings for (row, col) positions inside each image.

    Args:
        head_dim: int. Per-head dimension in vision attention. Must be even.
        theta: float. Base wavelength for rotary frequencies. Default 10000.
    """

    def __init__(self, head_dim, theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.theta = theta
        # Pre-compute inverse frequencies (not trainable).
        dim = head_dim // 2  # each spatial axis gets head_dim/2 dims
        idx = list(range(0, dim, 2))
        import math as _math

        self._inv_freq_vals = [
            1.0 / (_math.pow(theta, i / float(dim))) for i in idx
        ]
        self._inv_freq_len = len(self._inv_freq_vals)

    def get_freq_table(self, max_hw):
        """Produce a rotary frequency table of shape (max_hw, head_dim // 2).

        Called as a plain method (not a Keras layer call) to avoid input
        validation issues with integer arguments.

        Args:
            max_hw: int - max(H, W) of the image grid.
        Returns:
            Python list-based ops tensor of shape (max_hw, head_dim // 2).
        """
        inv_freq = ops.cast(
            ops.array(self._inv_freq_vals), "float32"
        )  # (dim//2,)
        positions = ops.cast(ops.arange(max_hw), "float32")  # (max_hw,)
        # (max_hw, dim//2)
        freqs = ops.einsum("i,j->ij", positions, inv_freq)
        # Duplicate to fill head_dim // 2: (max_hw, head_dim//2)
        freqs = ops.concatenate([freqs, freqs], axis=-1)
        return freqs

    def get_config(self):
        config = super().get_config()
        config.update({"head_dim": self.head_dim, "theta": self.theta})
        return config


class Qwen3_5VisionPatchEmbed(keras.layers.Layer):
    """3D Patch embedding using a single Conv3D layer.

    Converts raw pixel tensors into sequence of flattened patch features:
      (batch, T*H*W, hidden_size)

    The kernel and stride are both [temporal_patch_size, patch_size, patch_size]
    so there is no overlap between patches.

    Args:
        patch_size: int. Spatial patch size in pixels.
        temporal_patch_size: int. Temporal patch size in frames.
        in_channels: int. Number of input image channels (typically 3).
        hidden_size: int. Output embedding dimension per patch.
    """

    def __init__(
        self,
        patch_size,
        temporal_patch_size,
        in_channels,
        hidden_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

    def build(self, input_shape):
        kernel = [
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ]
        self.proj = keras.layers.Conv3D(
            filters=self.hidden_size,
            kernel_size=kernel,
            strides=kernel,
            use_bias=True,
            dtype=self.dtype_policy,
            name="proj",
        )
        # Build with an explicit shape so the layer is ready.
        self.proj.build(
            (
                None,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
                self.in_channels,
            )
        )
        self.built = True

    def call(self, pixel_values):
        """
        Args:
            pixel_values: Tensor of shape
                (total_patches, temporal_patch_size, patch_size, patch_size,
                 in_channels).
        Returns:
            Tensor of shape (total_patches, hidden_size).
        """
        # pixel_values: (N, T, H, W, C) where each patch is one "sample".
        x = self.proj(pixel_values)
        # After Conv3D with stride=kernel the spatial dims become 1x1x1.
        # Squeeze to (N, hidden_size).
        x = ops.squeeze(x, axis=(1, 2, 3))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "hidden_size": self.hidden_size,
            }
        )
        return config


class Qwen3_5VisionMLP(keras.layers.Layer):
    """Two-layer MLP with GELU activation used in each vision block."""

    def __init__(self, hidden_size, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def build(self, input_shape):
        self.fc1 = keras.layers.Dense(
            self.intermediate_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="fc1",
        )
        self.fc1.build(input_shape)
        self.fc2 = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="fc2",
        )
        self.fc2.build((input_shape[0], self.intermediate_size))
        self.built = True

    def call(self, x):
        x = self.fc1(x)
        x = ops.gelu(x, approximate=True)  # gelu_pytorch_tanh
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


class Qwen3_5VisionAttention(keras.layers.Layer):
    """Multi-head self-attention for vision tokens with 2D rotary embeddings.

    Args:
        hidden_size: int. Embed dimension.
        num_heads: int. Number of attention heads.
    """

    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self._inv_scale = 1.0 / math.sqrt(self.head_dim)

    def build(self, input_shape):
        self.qkv = keras.layers.Dense(
            self.hidden_size * 3,
            use_bias=True,
            dtype=self.dtype_policy,
            name="qkv",
        )
        self.qkv.build(input_shape)
        self.proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="proj",
        )
        self.proj.build((input_shape[0], self.hidden_size))
        self.built = True

    def _apply_rotary(self, x, cos_emb, sin_emb):
        """Apply rotary position embedding to x of
        shape (seq, heads, head_dim)."""
        # cos_emb / sin_emb shape: (seq, head_dim)
        cos_emb = ops.expand_dims(cos_emb, axis=1)  # (seq, 1, head_dim)
        sin_emb = ops.expand_dims(sin_emb, axis=1)

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rotated = ops.concatenate([-x2, x1], axis=-1)
        return (x * cos_emb) + (rotated * sin_emb)

    def call(self, x, position_embeddings, cu_seqlens=None):
        """
        Args:
            x: Tensor (seq_len, hidden_size).
            position_embeddings: tuple of (cos_emb, sin_emb) each
                (seq_len, head_dim).
            cu_seqlens: optional cumulative sequence lengths for variable
                length batching. Currently unused for simplicity; all tokens
                attend globally within the vision encoder (no causal mask).
        Returns:
            Tensor (seq_len, hidden_size).
        """
        seq_len = ops.shape(x)[0]

        qkv = self.qkv(x)  # (seq, 3 * hidden_size)
        qkv = ops.reshape(qkv, (seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        cos_emb, sin_emb = position_embeddings
        q = self._apply_rotary(q, cos_emb, sin_emb)
        k = self._apply_rotary(k, cos_emb, sin_emb)

        # Transpose for attention: (heads, seq, head_dim)
        q = ops.transpose(q, (1, 0, 2))
        k = ops.transpose(k, (1, 0, 2))
        v = ops.transpose(v, (1, 0, 2))

        # Scaled dot-product attention (full, no masking inside vision encoder)
        scores = ops.einsum("hid,hjd->hij", q, k) * self._inv_scale
        scores = ops.cast(scores, "float32")
        scores = ops.softmax(scores, axis=-1)
        scores = ops.cast(scores, self.compute_dtype)

        out = ops.einsum("hij,hjd->hid", scores, v)
        out = ops.transpose(out, (1, 0, 2))  # (seq, heads, head_dim)
        out = ops.reshape(out, (seq_len, self.hidden_size))
        out = self.proj(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "num_heads": self.num_heads}
        )
        return config


class Qwen3_5VisionBlock(keras.layers.Layer):
    """Single vision transformer block: LayerNorm → Attn → LN → MLP."""

    def __init__(self, hidden_size, num_heads, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size

    def build(self, input_shape):
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32", name="norm1"
        )
        self.norm1.build(input_shape)
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32", name="norm2"
        )
        self.norm2.build(input_shape)
        self.attn = Qwen3_5VisionAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.attn.build(input_shape)
        self.mlp = Qwen3_5VisionMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(input_shape)
        self.built = True

    def call(self, x, position_embeddings, cu_seqlens=None):
        # Vision always runs in float32 for stability.
        x = ops.cast(x, "float32")

        normed = self.norm1(x)
        normed = ops.cast(normed, self.compute_dtype)
        attn_out = self.attn(normed, position_embeddings, cu_seqlens)
        attn_out = ops.cast(attn_out, "float32")
        x = x + attn_out

        normed = self.norm2(x)
        normed = ops.cast(normed, self.compute_dtype)
        mlp_out = self.mlp(normed)
        mlp_out = ops.cast(mlp_out, "float32")
        x = x + mlp_out
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


class Qwen3_5VisionPatchMerger(keras.layers.Layer):
    """Spatial patch merger: groups spatial_merge_size² patches into one token.

    After the ViT, spatial tokens are grouped into non-overlapping
    spatial_merge_size×spatial_merge_size windows, and the hidden states
    within each window are concatenated and projected.

    Output hidden size: out_hidden_size (= text backbone hidden_dim).
    """

    def __init__(
        self, hidden_size, spatial_merge_size, out_hidden_size, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.merged_hidden = hidden_size * (spatial_merge_size**2)

    def build(self, input_shape):
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6, dtype="float32", name="ln_q"
        )
        self.norm.build((None, self.hidden_size))
        self.fc1 = keras.layers.Dense(
            self.merged_hidden,
            use_bias=True,
            dtype=self.dtype_policy,
            name="mlp_0",
        )
        self.fc1.build((None, self.merged_hidden))
        self.fc2 = keras.layers.Dense(
            self.out_hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="mlp_2",
        )
        self.fc2.build((None, self.merged_hidden))
        self.built = True

    def call(self, x):
        """
        Args:
            x: Tensor (total_tokens, hidden_size).
        Returns:
            Tensor (total_tokens // spatial_merge_size², out_hidden_size).
        """
        x = ops.cast(x, "float32")
        x = self.norm(x)
        # Group patches: reshape so spatial_merge_size² tokens merge
        total = ops.shape(x)[0]
        ms2 = self.spatial_merge_size**2
        n_merged = total // ms2
        x = ops.reshape(x, (n_merged, ms2 * self.hidden_size))
        x = ops.cast(x, self.compute_dtype)
        x = self.fc1(x)
        x = ops.gelu(x, approximate=True)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "spatial_merge_size": self.spatial_merge_size,
                "out_hidden_size": self.out_hidden_size,
            }
        )
        return config


class Qwen3_5VisionEncoder(keras.Model):
    """Vision encoder for Qwen3.5-VL (image-only, v1).

    Processes pre-extracted image patches through:
    1. Conv3D patch embedding
    2. Learnable absolute position embedding (bilinear interpolation)
    3. 2D rotary position embedding for attention
    4. ViT blocks (depth layers)
    5. Spatial patch merger → text hidden dimension

    All architecture parameters are explicit arguments so the class works for
    any model variant (0.8B–27B).

    Args:
        depth: int. Number of ViT transformer blocks.
        hidden_size: int. ViT hidden dimension.
        num_heads: int. Number of attention heads in ViT.
        intermediate_size: int. FFN intermediate dimension.
        in_channels: int. Input image channels (default 3).
        patch_size: int. Spatial patch size in pixels.
        temporal_patch_size: int. Temporal patch size.
        spatial_merge_size: int. Spatial merge downsampling factor.
        out_hidden_size: int. Projection output dim (= text backbone hidden).
        num_position_embeddings: int. Max absolute position embeddings.
        dtype: compute dtype.
    """

    def __init__(
        self,
        depth,
        hidden_size,
        num_heads,
        intermediate_size,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=None,
        num_position_embeddings=2304,
        dtype=None,
        **kwargs,
    ):
        # Always run the vision encoder in float32 for numerical stability.
        if dtype is not None and dtype != "float32":
            dtype = "float32"

        super().__init__(dtype=dtype, **kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size or hidden_size
        self.num_position_embeddings = num_position_embeddings

        head_dim = hidden_size // num_heads

        # Layers
        self.patch_embed = Qwen3_5VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            dtype="float32",
            name="patch_embed",
        )

        # Absolute position embedding table (bilinear interpolated).
        self.pos_embed = keras.layers.Embedding(
            num_position_embeddings,
            hidden_size,
            dtype="float32",
            name="pos_embed",
        )

        self.num_grid_per_side = int(math.sqrt(num_position_embeddings))

        # Rotary position embedding for 2D spatial positions.
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(
            head_dim=head_dim,
            dtype="float32",
            name="rotary_pos_emb",
        )

        # ViT blocks.
        self.blocks = [
            Qwen3_5VisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dtype="float32",
                name=f"blocks_{i}",
            )
            for i in range(depth)
        ]

        # Spatial merger.
        self.merger = Qwen3_5VisionPatchMerger(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=self.out_hidden_size,
            dtype="float32",
            name="merger",
        )

    def _fast_pos_embed_interpolate(self, grid_thw):
        """Bilinear interpolation of absolute position embeddings.

        For each image, interpolates the Embedding table to cover an H×W grid,
        then repeats along the temporal axis.

        Args:
            grid_thw: int tensor (num_images, 3) with [T, H, W] per image.
        Returns:
            Tensor (total_tokens, hidden_size).
        """
        all_embeds = []
        n = grid_thw.shape[0] if hasattr(grid_thw, "shape") else len(grid_thw)
        for idx in range(n):
            t = int(grid_thw[idx][0])
            h = int(grid_thw[idx][1])
            w = int(grid_thw[idx][2])
            gs = self.num_grid_per_side

            # Row / col indices in the embedding table (float for interp).
            h_idxs = ops.linspace(0.0, float(gs - 1), h)
            w_idxs = ops.linspace(0.0, float(gs - 1), w)

            h_floor = ops.cast(ops.floor(h_idxs), "int32")
            w_floor = ops.cast(ops.floor(w_idxs), "int32")
            h_ceil = ops.clip(h_floor + 1, 0, gs - 1)
            w_ceil = ops.clip(w_floor + 1, 0, gs - 1)

            dh = h_idxs - ops.cast(h_floor, "float32")
            dw = w_idxs - ops.cast(w_floor, "float32")

            # Four-corner indices in the flat embedding table.
            base_hf = h_floor * gs
            base_hc = h_ceil * gs

            # Bilinear weights (h, w)
            w00 = ops.outer(1.0 - dh, 1.0 - dw)
            w01 = ops.outer(1.0 - dh, dw)
            w10 = ops.outer(dh, 1.0 - dw)
            w11 = ops.outer(dh, dw)

            # Indices (h, w)
            idx00 = ops.cast(
                ops.expand_dims(base_hf, 1) + ops.expand_dims(w_floor, 0),
                "int32",
            )
            idx01 = ops.cast(
                ops.expand_dims(base_hf, 1) + ops.expand_dims(w_ceil, 0),
                "int32",
            )
            idx10 = ops.cast(
                ops.expand_dims(base_hc, 1) + ops.expand_dims(w_floor, 0),
                "int32",
            )
            idx11 = ops.cast(
                ops.expand_dims(base_hc, 1) + ops.expand_dims(w_ceil, 0),
                "int32",
            )

            # Lookup and bilinear blend.
            e00 = self.pos_embed(ops.reshape(idx00, (-1,)))
            e01 = self.pos_embed(ops.reshape(idx01, (-1,)))
            e10 = self.pos_embed(ops.reshape(idx10, (-1,)))
            e11 = self.pos_embed(ops.reshape(idx11, (-1,)))

            w00_flat = ops.reshape(w00, (-1, 1))
            w01_flat = ops.reshape(w01, (-1, 1))
            w10_flat = ops.reshape(w10, (-1, 1))
            w11_flat = ops.reshape(w11, (-1, 1))

            patch_embed = (
                e00 * w00_flat
                + e01 * w01_flat
                + e10 * w10_flat
                + e11 * w11_flat
            )  # (h*w, hidden_size)

            # Repeat across temporal frames.
            patch_embed = ops.tile(
                ops.expand_dims(patch_embed, 0), [t, 1, 1]
            )  # (T, H*W, H)
            ms = self.spatial_merge_size
            patch_embed = ops.reshape(
                patch_embed, (t, h // ms, ms, w // ms, ms, self.hidden_size)
            )
            patch_embed = ops.transpose(patch_embed, (0, 1, 3, 2, 4, 5))
            patch_embed = ops.reshape(patch_embed, (-1, self.hidden_size))
            all_embeds.append(patch_embed)

        return ops.concatenate(all_embeds, axis=0)

    def _rot_pos_emb(self, grid_thw):
        """Produce 2D rotary position embeddings for all image patches.

        Args:
            grid_thw: list/tensor (num_images, 3) with [T, H, W] per image.
        Returns:
            cos_emb, sin_emb: each (total_tokens, head_dim).
        """
        ms = self.spatial_merge_size
        n = grid_thw.shape[0] if hasattr(grid_thw, "shape") else len(grid_thw)
        max_hw = max(
            max(int(grid_thw[i][1]), int(grid_thw[i][2])) for i in range(n)
        )
        # Use plain method (not Keras layer call) to avoid integer input
        # validation.
        freq_table = self.rotary_pos_emb.get_freq_table(
            max_hw
        )  # (max_hw, head_dim//2)

        all_embeds = []
        for idx in range(n):
            t_val = int(grid_thw[idx][0])
            h_val = int(grid_thw[idx][1])
            w_val = int(grid_thw[idx][2])

            merged_h = h_val // ms
            merged_w = w_val // ms

            # Build row/col indices maintaining merge order.
            brows = ops.arange(merged_h)
            bcols = ops.arange(merged_w)
            intra_r = ops.arange(ms)
            intra_c = ops.arange(ms)

            # (merged_h, merged_w, ms, ms)
            row_idx = ops.reshape(
                brows, (merged_h, 1, 1, 1)
            ) * ms + ops.reshape(intra_r, (1, 1, ms, 1))
            col_idx = ops.reshape(
                bcols, (1, merged_w, 1, 1)
            ) * ms + ops.reshape(intra_c, (1, 1, 1, ms))
            row_idx = ops.broadcast_to(row_idx, (merged_h, merged_w, ms, ms))
            col_idx = ops.broadcast_to(col_idx, (merged_h, merged_w, ms, ms))
            row_idx = ops.reshape(row_idx, (-1,))
            col_idx = ops.reshape(col_idx, (-1,))

            # Lookup frequencies.
            row_freqs = ops.take(freq_table, row_idx, axis=0)  # (h*w, hd//2)
            col_freqs = ops.take(freq_table, col_idx, axis=0)

            # Interleave as (row, col) → (h*w, head_dim)
            spatial_emb = ops.concatenate([row_freqs, col_freqs], axis=-1)

            # Repeat for temporal frames.
            if t_val > 1:
                spatial_emb = ops.tile(
                    ops.expand_dims(spatial_emb, 0), [t_val, 1, 1]
                )
                n_hw = ops.shape(spatial_emb)[1]
                h_dim = ops.shape(spatial_emb)[2]
                spatial_emb = ops.reshape(spatial_emb, (t_val * n_hw, h_dim))

            all_embeds.append(spatial_emb)

        rotary = ops.concatenate(all_embeds, axis=0)
        cos_emb = ops.cos(rotary)
        sin_emb = ops.sin(rotary)
        return cos_emb, sin_emb

    def call(self, pixel_values, grid_thw):
        """Forward pass through vision encoder.

        Args:
            pixel_values: Tensor (total_patches, temporal_patch_size,
                          patch_size, patch_size, in_channels).
                          Pre-extracted patches for all images concatenated.
            grid_thw: int tensor or list (num_images, 3) — [T, H, W] per image
                      **in units of patches after merging** (i.e. after
                      dividing by patch_size; temporal dim 1 for images).
        Returns:
            Tensor (total_merged_tokens, out_hidden_size).
        """
        # 1. Patch embedding.
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = ops.cast(hidden_states, "float32")

        # 2. Absolute position embedding.
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 3. Rotary position embedding.
        cos_emb, sin_emb = self._rot_pos_emb(grid_thw)
        position_embeddings = (cos_emb, sin_emb)

        # 4. ViT blocks.
        for blk in self.blocks:
            hidden_states = blk(hidden_states, position_embeddings)

        # 5. Patch merger.
        merged = self.merger(hidden_states)
        return merged

    def compute_output_spec(self, pixel_values, grid_thw=None):
        """Return the output shape spec without running forward pass.

        Needed because the backbone wraps the encoder inside a Keras
        functional model and Keras must infer the output shape at
        construction time.

        Returns:
            KerasTensor with shape ``(None, out_hidden_size)`` and
            dtype ``float32``.
        """
        import keras

        return keras.KerasTensor(
            shape=(None, self.out_hidden_size),
            dtype="float32",
        )

    def get_config(self):
        return {
            "depth": self.depth,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "out_hidden_size": self.out_hidden_size,
            "num_position_embeddings": self.num_position_embeddings,
        }
