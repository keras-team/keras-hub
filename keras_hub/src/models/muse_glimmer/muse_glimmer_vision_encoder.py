import math

import keras
import numpy as np
from keras import ops


class MuseGlimmerVisionRotaryEmbedding(keras.layers.Layer):
    """Axial 2D rotary position embedding for the vision tower.

    The layer computes frequencies independently for each spatial axis.
    The layer recomposes frequencies as `[freq_w, freq_h, freq_w, freq_h]`.
    The order matches HuggingFace MuseGlimmer.

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

    def get_freqs(self, positions):
        inv_freq = ops.cast(ops.array(self._inv_freq_vals), "float32")
        positions = ops.cast(positions, "float32")
        return ops.einsum("ni,j->nij", positions, inv_freq)

    def recomposition_frequencies(self, freq):
        # freq shape is (seq, 2, head_dim // 4).
        freq_w, freq_h = freq[..., 0, :], freq[..., 1, :]
        freq_wh = ops.concatenate([freq_w, freq_h], axis=-1)
        return ops.concatenate([freq_wh, freq_wh], axis=-1)

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
        """Interpolate the learned position table for each patch.

        The interpolation uses half-pixel coordinates and zero padding.
        The interpolation matches HuggingFace `grid_sample` behavior.
        """
        gh, gw = self.pos_emb_height, self.pos_emb_width
        all_embeds = []
        for t_val, h_val, w_val in grid_thw:
            t_val, h_val, w_val = int(t_val), int(h_val), int(w_val)
            h_index = ops.cast(ops.arange(h_val), "float32")
            w_index = ops.cast(ops.arange(w_val), "float32")
            h_source = (h_index + 0.5) * gh / h_val - 0.5
            w_source = (w_index + 0.5) * gw / w_val - 0.5
            h_floor = ops.floor(h_source)
            w_floor = ops.floor(w_source)
            offsets = ops.cast(ops.arange(2), "float32")

            h_raw_taps = ops.expand_dims(h_floor, -1) + offsets
            w_raw_taps = ops.expand_dims(w_floor, -1) + offsets
            h_taps = ops.cast(ops.clip(h_raw_taps, 0, gh - 1), "int32")
            w_taps = ops.cast(ops.clip(w_raw_taps, 0, gw - 1), "int32")

            h_distance = ops.abs(
                ops.expand_dims(h_source, -1)
                - ops.expand_dims(h_floor, -1)
                - offsets
            )
            w_distance = ops.abs(
                ops.expand_dims(w_source, -1)
                - ops.expand_dims(w_floor, -1)
                - offsets
            )
            h_weights = ops.clip(1.0 - h_distance, 0.0, 1.0)
            w_weights = ops.clip(1.0 - w_distance, 0.0, 1.0)
            h_weights = h_weights * ops.cast(
                (h_raw_taps >= 0) & (h_raw_taps <= gh - 1), "float32"
            )
            w_weights = w_weights * ops.cast(
                (w_raw_taps >= 0) & (w_raw_taps <= gw - 1), "float32"
            )

            corner_indices = ops.reshape(
                ops.expand_dims(ops.expand_dims(h_taps, 1), -1) * gw
                + ops.expand_dims(ops.expand_dims(w_taps, 0), 2),
                (-1, 4),
            )
            corner_weights = ops.reshape(
                ops.expand_dims(ops.expand_dims(h_weights, 1), -1)
                * ops.expand_dims(ops.expand_dims(w_weights, 0), 2),
                (-1, 4),
            )
            patch_embed = ops.sum(
                self.position_embedding_table(corner_indices)
                * ops.expand_dims(corner_weights, -1),
                axis=1,
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
        x_dtype = x.dtype
        x = ops.cast(x, "float32")
        cos_emb = ops.cast(cos_emb, "float32")
        sin_emb = ops.cast(sin_emb, "float32")
        x = (x * cos_emb) + (_rotate_half(x) * sin_emb)
        return ops.cast(x, x_dtype)

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
            sc = ops.matmul(q_c, ops.transpose(k_c, (0, 2, 1)))
            sc = sc * self._inv_scale
            sc = ops.cast(
                ops.softmax(ops.cast(sc, "float32"), axis=-1), v.dtype
            )
            out_chunks.append(ops.matmul(sc, v_c))
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

    The helper pads ragged windows before it removes padding indices.
    """
    all_blocks = []
    offset = 0
    for frames, height, width in grid_thw:
        frames, height, width = int(frames), int(height), int(width)
        padded_height = int(np.ceil(height / window_patches) * window_patches)
        padded_width = int(np.ceil(width / window_patches) * window_patches)
        num_window_rows = padded_height // window_patches
        num_window_cols = padded_width // window_patches

        grid = np.full(
            (frames, padded_height, padded_width), -1, dtype=np.int64
        )
        frame_grid = np.arange(height * width).reshape(height, width)
        grid[:, :height, :width] = frame_grid
        frame_offsets = np.arange(frames)[:, None, None] * height * width
        grid = np.where(grid >= 0, grid + frame_offsets, -1)

        blocks = grid.reshape(
            frames,
            num_window_rows,
            window_patches,
            num_window_cols,
            window_patches,
        )
        blocks = blocks.transpose(0, 1, 3, 2, 4).reshape(
            -1, window_patches * window_patches
        )
        for block in blocks:
            block = block[block >= 0] + offset
            all_blocks.append(block)
        offset += frames * height * width

    window_index = np.concatenate(all_blocks, axis=0)
    window_sizes = [block.size for block in all_blocks]
    cu_window_seqlens = np.concatenate([[0], np.cumsum(window_sizes)])
    return window_index, cu_window_seqlens.astype(np.int32)


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
        all_freqs = []
        for t_val, h_val, w_val in grid_thw:
            t_val, h_val, w_val = int(t_val), int(h_val), int(w_val)
            row_idx = np.repeat(np.arange(h_val), w_val) + 1
            col_idx = np.tile(np.arange(w_val), h_val) + 1
            # HuggingFace flips raster coordinates and adds one before RoPE.
            positions = ops.array(
                np.stack([col_idx, row_idx], axis=-1), dtype="int32"
            )
            if t_val > 1:
                positions = ops.tile(positions, (t_val, 1))
            all_freqs.append(self.rotary_pos_emb.get_freqs(positions))
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
        if grid_np.ndim == 1:
            grid_np = grid_np[np.newaxis, :]

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

        lengths = []
        for t_val, h_val, w_val in grid_np:
            lengths.extend([int(h_val) * int(w_val)] * int(t_val))
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
