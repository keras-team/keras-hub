import math

import keras
import numpy as np
from keras import ops


class MuseGlimmerVisionRotaryEmbedding(keras.layers.Layer):
    """Axial (row/col) 2D rotary position embedding for the vision tower.

    Frequencies are computed independently per spatial axis (H, W) and then
    recomposed as `[freq_h, freq_w, freq_h, freq_w]` — this specific concat
    order (not a plain per-axis half-split) must be preserved, matching
    `MuseGlimmerVisionRotaryEmbedding.recomposition_frequencies` in
    `modeling_muse_glimmer.py`.

    Args:
        head_dim: int. Per-head dimension in vision attention.
        theta: float. RoPE base wavelength. Defaults to `10000.0`.
    """

    def __init__(self, head_dim, theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.theta = theta
        spatial_dim = head_dim // 2
        idx = list(range(0, spatial_dim, 2))
        self._inv_freq_vals = [
            1.0 / (theta ** (i / float(spatial_dim))) for i in idx
        ]

    def get_freq_table(self, max_grid):
        inv_freq = ops.cast(ops.array(self._inv_freq_vals), "float32")
        positions = ops.cast(ops.arange(max_grid), "float32")
        return ops.einsum("i,j->ij", positions, inv_freq)

    def recomposition_frequencies(self, freq):
        # freq: (seq, head_dim // 4) per axis -> (seq, head_dim).
        freq_h, freq_w = freq[..., 0, :], freq[..., 1, :]
        freq_hw = ops.concatenate([freq_h, freq_w], axis=-1)
        return ops.concatenate([freq_hw, freq_hw], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"head_dim": self.head_dim, "theta": self.theta})
        return config


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.concatenate([-x2, x1], axis=-1)


class MuseGlimmerVisionPatchEmbedder(keras.layers.Layer):
    """Patch embedding + bilinearly-interpolated learned position embedding.

    Args:
        hidden_size: int. Vision tower hidden dimension.
        pos_emb_height: int. Height of the learned position embedding grid.
        pos_emb_width: int. Width of the learned position embedding grid.
    """

    def __init__(self, hidden_size, pos_emb_height, pos_emb_width, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width

    def build(self, input_shape):
        self.patch_embedding = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype_policy,
            name="patch_embedding",
        )
        self.patch_embedding.build(input_shape)
        self.position_embedding_table = keras.layers.Embedding(
            self.pos_emb_height * self.pos_emb_width,
            self.hidden_size,
            dtype=self.dtype_policy,
            name="position_embedding_table",
        )
        self.position_embedding_table.build((None,))
        self.built = True

    def _bilinear_position_embeddings(self, grid_thw):
        """Bilinearly interpolate the learned position table per patch.

        `grid_thw` is a concrete (non-symbolic) numpy array — the vision
        tower runs imperatively outside the Functional graph, matching the
        convention used by `Qwen3_5VisionEncoder`.
        """
        gh, gw = self.pos_emb_height, self.pos_emb_width
        all_embeds = []
        for t_val, h_val, w_val in grid_thw:
            t_val, h_val, w_val = int(t_val), int(h_val), int(w_val)
            h_idxs = ops.linspace(0.0, float(gh - 1), h_val)
            w_idxs = ops.linspace(0.0, float(gw - 1), w_val)
            h_floor = ops.cast(ops.floor(h_idxs), "int32")
            w_floor = ops.cast(ops.floor(w_idxs), "int32")
            h_ceil = ops.clip(h_floor + 1, 0, gh - 1)
            w_ceil = ops.clip(w_floor + 1, 0, gw - 1)
            dh = h_idxs - ops.cast(h_floor, "float32")
            dw = w_idxs - ops.cast(w_floor, "float32")

            base_hf = h_floor * gw
            base_hc = h_ceil * gw

            w00 = ops.expand_dims(1.0 - dh, -1) * ops.expand_dims(1.0 - dw, 0)
            w01 = ops.expand_dims(1.0 - dh, -1) * ops.expand_dims(dw, 0)
            w10 = ops.expand_dims(dh, -1) * ops.expand_dims(1.0 - dw, 0)
            w11 = ops.expand_dims(dh, -1) * ops.expand_dims(dw, 0)

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

            e00 = self.position_embedding_table(ops.reshape(idx00, (-1,)))
            e01 = self.position_embedding_table(ops.reshape(idx01, (-1,)))
            e10 = self.position_embedding_table(ops.reshape(idx10, (-1,)))
            e11 = self.position_embedding_table(ops.reshape(idx11, (-1,)))

            patch_embed = (
                e00 * ops.reshape(w00, (-1, 1))
                + e01 * ops.reshape(w01, (-1, 1))
                + e10 * ops.reshape(w10, (-1, 1))
                + e11 * ops.reshape(w11, (-1, 1))
            )
            patch_embed = ops.tile(
                ops.expand_dims(patch_embed, 0), [t_val, 1, 1]
            )
            patch_embed = ops.reshape(patch_embed, (-1, self.hidden_size))
            all_embeds.append(patch_embed)
        return ops.concatenate(all_embeds, axis=0)

    def call(self, pixel_values, grid_thw):
        embeddings = self.patch_embedding(pixel_values)
        pos_embeds = self._bilinear_position_embeddings(grid_thw)
        return embeddings + ops.cast(pos_embeds, embeddings.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "pos_emb_height": self.pos_emb_height,
                "pos_emb_width": self.pos_emb_width,
            }
        )
        return config


class MuseGlimmerVisionAttention(keras.layers.Layer):
    """Standard multi-head self-attention with axial 2D RoPE, no GQA."""

    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self._inv_scale = 1.0 / math.sqrt(self.head_dim)

    def build(self, input_shape):
        self.q_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="proj",
        )
        self.proj.build((input_shape[0], self.hidden_size))
        self.built = True

    def _apply_rotary(self, x, cos_emb, sin_emb):
        cos_emb = ops.expand_dims(cos_emb, axis=1)
        sin_emb = ops.expand_dims(sin_emb, axis=1)
        return (x * cos_emb) + (_rotate_half(x) * sin_emb)

    def call(self, x, position_embeddings, cu_seqlens):
        seq_len = ops.shape(x)[0]
        q = ops.reshape(
            self.q_proj(x), (seq_len, self.num_heads, self.head_dim)
        )
        k = ops.reshape(
            self.k_proj(x), (seq_len, self.num_heads, self.head_dim)
        )
        v = ops.reshape(
            self.v_proj(x), (seq_len, self.num_heads, self.head_dim)
        )

        cos_emb, sin_emb = position_embeddings
        q = self._apply_rotary(q, cos_emb, sin_emb)
        k = self._apply_rotary(k, cos_emb, sin_emb)

        q = ops.transpose(q, (1, 0, 2))
        k = ops.transpose(k, (1, 0, 2))
        v = ops.transpose(v, (1, 0, 2))

        cu_np = np.array(cu_seqlens)
        out_chunks = []
        for ci in range(len(cu_np) - 1):
            s, e = int(cu_np[ci]), int(cu_np[ci + 1])
            q_c, k_c, v_c = q[:, s:e, :], k[:, s:e, :], v[:, s:e, :]
            sc = ops.einsum("hid,hjd->hij", q_c, k_c) * self._inv_scale
            sc = ops.cast(
                ops.softmax(ops.cast(sc, "float32"), axis=-1), v.dtype
            )
            out_chunks.append(ops.einsum("hij,hjd->hid", sc, v_c))
        out = ops.concatenate(out_chunks, axis=1)

        out = ops.transpose(out, (1, 0, 2))
        out = ops.reshape(out, (seq_len, self.hidden_size))
        return self.proj(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_size": self.hidden_size, "num_heads": self.num_heads}
        )
        return config


class MuseGlimmerVisionMLP(keras.layers.Layer):
    """Two-layer MLP, no gating: `fc2(gelu(fc1(x)))`."""

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
            self.hidden_size, use_bias=True, dtype=self.dtype_policy, name="fc2"
        )
        self.fc2.build((input_shape[0], self.intermediate_size))
        self.built = True

    def call(self, x):
        return self.fc2(keras.activations.gelu(self.fc1(x), approximate=False))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
            }
        )
        return config


class MuseGlimmerVisionEncoderLayer(keras.layers.Layer):
    """Pre-LN vision transformer block.

    LN -> attn -> residual, then LN -> mlp -> residual.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps

    def build(self, input_shape):
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, dtype=self.dtype_policy, name="norm1"
        )
        self.norm1.build(input_shape)
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, dtype=self.dtype_policy, name="norm2"
        )
        self.norm2.build(input_shape)
        self.attn = MuseGlimmerVisionAttention(
            self.hidden_size,
            self.num_heads,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.attn.build(input_shape)
        self.mlp = MuseGlimmerVisionMLP(
            self.hidden_size,
            self.intermediate_size,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(input_shape)
        self.built = True

    def call(self, x, position_embeddings, cu_seqlens):
        x = x + self.attn(self.norm1(x), position_embeddings, cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


def get_vision_pixel_shuffle_index(grid_thw, merge_size):
    """Permutation grouping `merge_size x merge_size` spatial blocks.

    Matches `get_vision_pixel_shuffle_index` in `modeling_muse_glimmer.py`
    exactly (frame-aware for `frames > 1`).
    """
    indices = []
    offset = 0
    for frames, height, width in grid_thw:
        frames, height, width = int(frames), int(height), int(width)
        permutation = np.arange(height * width)
        permutation = permutation.reshape(
            height // merge_size, merge_size, width // merge_size, merge_size
        )
        permutation = permutation.transpose(0, 2, 1, 3).reshape(-1)
        if frames > 1:
            frame_offsets = (np.arange(frames) * height * width).reshape(
                frames, 1
            )
            permutation = (permutation[None, :] + frame_offsets).reshape(-1)
        indices.append(permutation + offset)
        offset += frames * height * width
    return np.concatenate(indices, axis=0)


def _get_window_index(grid_thw, window_patches):
    """Window-index permutation for alternating window/full attention.

    Groups each image's (pre-merge) patch grid into
    `window_patches x window_patches` windows and returns a permutation so
    that patches within a window are contiguous, plus the cumulative
    sequence lengths of each window (`cu_window_seqlens`).

    This is a best-effort reconstruction of `get_vision_window_index` (the
    HF utility body lives in `vision_utils`, not surfaced by the gatherer)
    following the well-documented Qwen2.5-VL-style windowing scheme
    described in the migration report; numerics must still be validated
    against real weights.
    """
    indices = []
    window_lengths = [0]
    offset = 0
    for frames, height, width in grid_thw:
        frames, height, width = int(frames), int(height), int(width)
        grid = np.arange(height * width).reshape(height, width)
        for t in range(frames):
            frame_offset = t * height * width
            for h0 in range(0, height, window_patches):
                for w0 in range(0, width, window_patches):
                    block = grid[
                        h0 : h0 + window_patches, w0 : w0 + window_patches
                    ].reshape(-1)
                    indices.append(block + frame_offset + offset)
                    window_lengths.append(window_lengths[-1] + block.size)
        offset += frames * height * width
    window_index = np.concatenate(indices, axis=0)
    cu_window_seqlens = np.array(window_lengths, dtype=np.int32)
    return window_index, cu_window_seqlens


class MuseGlimmerVisionEncoder(keras.Model):
    """MuseGlimmer's ViT-style perception encoder.

    Processes pre-extracted, pre-flattened image patches through patch
    embedding + bilinear position embeddings, alternating window/full
    attention (via index permutation, not a windowed attention mask), axial
    2D RoPE, and pixel-shuffle patch merging.

    Args:
        num_layers: int. Number of transformer blocks.
        hidden_size: int. ViT hidden dimension.
        num_heads: int. Number of attention heads.
        intermediate_size: int. MLP intermediate dimension.
        patch_size: int. Spatial patch size in pixels.
        patch_temporal: int. Temporal patch size in frames.
        merge_size: int. Spatial merge (pixel-shuffle) factor.
        pos_emb_height: int. Learned position-embedding grid height.
        pos_emb_width: int. Learned position-embedding grid width.
        rope_theta: float. Axial RoPE base wavelength. Defaults to `10000.0`.
        layer_norm_eps: float. Epsilon for all LayerNorms. Defaults to
            `1e-5`.
        layer_types: list of str or `None`. Per-layer `"window_attention"`
            or `"full_attention"`. Defaults to every 4th layer (and the
            last) being full attention.
        out_hidden_size: int or `None`. Output dim after pixel-shuffle
            merge; defaults to `hidden_size * merge_size ** 2`.
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_heads,
        intermediate_size,
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=32,
        pos_emb_width=32,
        rope_theta=10000.0,
        layer_norm_eps=1e-5,
        layer_types=None,
        out_hidden_size=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.rope_theta = rope_theta
        self.layer_norm_eps = layer_norm_eps
        if layer_types is None:
            layer_types = [
                "full_attention"
                if (i + 1) % 4 == 0 or i == num_layers - 1
                else "window_attention"
                for i in range(num_layers)
            ]
        self.layer_types = layer_types
        self.out_hidden_size = out_hidden_size or (hidden_size * merge_size**2)
        self.window_size = pos_emb_height * patch_size

        head_dim = hidden_size // num_heads
        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(
            hidden_size,
            pos_emb_height,
            pos_emb_width,
            dtype=self.dtype_policy,
            name="patch_embedder",
        )
        self.ln_pre = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            dtype=self.dtype_policy,
            name="ln_pre",
        )
        self.rotary_pos_emb = MuseGlimmerVisionRotaryEmbedding(
            head_dim,
            theta=rope_theta,
            dtype=self.dtype_policy,
            name="rotary_pos_emb",
        )
        self.blocks = [
            MuseGlimmerVisionEncoderLayer(
                hidden_size,
                num_heads,
                intermediate_size,
                layer_norm_eps=layer_norm_eps,
                dtype=self.dtype_policy,
                name=f"layers_{i}",
            )
            for i in range(num_layers)
        ]
        self.ln_post = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            dtype=self.dtype_policy,
            name="ln_post",
        )

    def build(self, input_shape=None):
        patch_dim = self.patch_temporal * 3 * self.patch_size**2
        if not self.patch_embedder.built:
            self.patch_embedder.build((None, patch_dim))
        if not self.ln_pre.built:
            self.ln_pre.build((None, self.hidden_size))
        for layer in self.blocks:
            if not layer.built:
                layer.build((None, self.hidden_size))
        if not self.ln_post.built:
            self.ln_post.build((None, self.hidden_size))
        super().build(input_shape)

    def _rot_pos_emb(self, grid_thw):
        max_grid = max(max(int(t[1]), int(t[2])) for t in grid_thw)
        freq_table = self.rotary_pos_emb.get_freq_table(max_grid)

        all_freqs = []
        for t_val, h_val, w_val in grid_thw:
            t_val, h_val, w_val = int(t_val), int(h_val), int(w_val)
            row_idx = np.repeat(np.arange(h_val), w_val)
            col_idx = np.tile(np.arange(w_val), h_val)
            row_freqs = ops.take(freq_table, row_idx, axis=0)
            col_freqs = ops.take(freq_table, col_idx, axis=0)
            per_patch = ops.stack([row_freqs, col_freqs], axis=1)
            if t_val > 1:
                per_patch = ops.tile(
                    ops.expand_dims(per_patch, 0), [t_val, 1, 1, 1]
                )
                per_patch = ops.reshape(
                    per_patch, (t_val * h_val * w_val, 2, -1)
                )
            all_freqs.append(per_patch)
        freqs = ops.concatenate(all_freqs, axis=0)
        cos = self.rotary_pos_emb.recomposition_frequencies(ops.cos(freqs))
        sin = self.rotary_pos_emb.recomposition_frequencies(ops.sin(freqs))
        return cos, sin

    def call(self, pixel_values, grid_thw):
        batched = len(ops.shape(pixel_values)) == 3
        if batched:
            patch_dim = ops.shape(pixel_values)[-1]
            pixel_values = ops.reshape(pixel_values, (-1, patch_dim))
            grid_thw = ops.reshape(grid_thw, (-1, 3))

        num_patches = ops.shape(pixel_values)[0]
        if num_patches == 0:
            empty = ops.zeros((0, self.out_hidden_size))
            return ops.expand_dims(empty, axis=0) if batched else empty

        grid_np = np.array(ops.convert_to_numpy(grid_thw))

        hidden_states = self.patch_embedder(pixel_values, grid_np)
        hidden_states = self.ln_pre(hidden_states)

        window_patches = max(self.window_size // self.patch_size, 1)
        window_index, cu_window_seqlens = _get_window_index(
            grid_np, window_patches
        )
        hidden_states = ops.take(hidden_states, window_index, axis=0)

        cos, sin = self._rot_pos_emb(grid_np)
        cos = ops.take(cos, window_index, axis=0)
        sin = ops.take(sin, window_index, axis=0)

        lengths = [int(t) * int(h) * int(w) for t, h, w in grid_np]
        cu_seqlens = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)

        for layer, layer_type in zip(self.blocks, self.layer_types):
            seqlens = (
                cu_seqlens
                if layer_type == "full_attention"
                else cu_window_seqlens
            )
            hidden_states = layer(hidden_states, (cos, sin), seqlens)

        reverse_indices = np.argsort(window_index)
        hidden_states = ops.take(hidden_states, reverse_indices, axis=0)
        hidden_states = self.ln_post(hidden_states)

        shuffle_index = get_vision_pixel_shuffle_index(grid_np, self.merge_size)
        hidden_states = ops.take(hidden_states, shuffle_index, axis=0)
        factor = self.merge_size
        dim = self.hidden_size
        hidden_states = ops.reshape(hidden_states, (-1, factor * factor, dim))
        hidden_states = ops.transpose(hidden_states, (0, 2, 1))
        hidden_states = ops.reshape(hidden_states, (-1, dim * factor * factor))

        if batched:
            hidden_states = ops.expand_dims(hidden_states, axis=0)
        return hidden_states

    def compute_output_spec(self, pixel_values, grid_thw=None):
        if len(pixel_values.shape) == 3:
            return keras.KerasTensor(
                shape=(pixel_values.shape[0], None, self.out_hidden_size),
                dtype="float32",
            )
        return keras.KerasTensor(
            shape=(None, self.out_hidden_size), dtype="float32"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "patch_size": self.patch_size,
                "patch_temporal": self.patch_temporal,
                "merge_size": self.merge_size,
                "pos_emb_height": self.pos_emb_height,
                "pos_emb_width": self.pos_emb_width,
                "rope_theta": self.rope_theta,
                "layer_norm_eps": self.layer_norm_eps,
                "layer_types": self.layer_types,
                "out_hidden_size": self.out_hidden_size,
            }
        )
        return config
