import math

import keras
import numpy as np
from keras import ops


def _per_patch_coordinates(grid_thw, num_patches):
    """Per-patch `(row, col, height, width, frame_id, frame_start)` via
    `ops.searchsorted` + gather, no Python loop.
    """
    if len(ops.shape(grid_thw)) == 1:
        grid_thw = ops.expand_dims(grid_thw, axis=0)
    grid_thw = ops.cast(grid_thw, "int32")
    t, h, w = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

    lengths = t * h * w
    cumulative = ops.cumsum(lengths)
    starts = ops.concatenate(
        [ops.zeros((1,), dtype=cumulative.dtype), cumulative[:-1]]
    )
    positions = ops.arange(num_patches, dtype="int32")
    image_id = ops.cast(
        ops.searchsorted(cumulative, positions, side="right"), "int32"
    )
    local_offset = positions - ops.take(starts, image_id)

    h_per_patch = ops.take(h, image_id)
    w_per_patch = ops.take(w, image_id)
    hw_per_patch = h_per_patch * w_per_patch
    offset_in_frame = ops.mod(local_offset, hw_per_patch)
    frame_within_image = ops.floor_divide(local_offset, hw_per_patch)

    row = ops.floor_divide(offset_in_frame, w_per_patch)
    col = ops.mod(offset_in_frame, w_per_patch)

    frame_starts = ops.cumsum(t) - t
    frame_id = ops.take(frame_starts, image_id) + frame_within_image
    frame_start = ops.take(starts, image_id) + frame_within_image * (
        hw_per_patch
    )

    return row, col, h_per_patch, w_per_patch, frame_id, frame_start


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

    def _bilinear_position_embeddings(self, grid_thw, num_patches):
        """Interpolate the learned position table for each patch.

        The interpolation uses half-pixel coordinates and zero padding.
        The interpolation matches HuggingFace `grid_sample` behavior.
        """
        gh, gw = self.pos_emb_height, self.pos_emb_width
        row, col, h_per_patch, w_per_patch, _, _ = _per_patch_coordinates(
            grid_thw, num_patches
        )
        row = ops.cast(row, "float32")
        col = ops.cast(col, "float32")
        h_per_patch = ops.cast(h_per_patch, "float32")
        w_per_patch = ops.cast(w_per_patch, "float32")

        h_source = (row + 0.5) * gh / h_per_patch - 0.5
        w_source = (col + 0.5) * gw / w_per_patch - 0.5
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
            ops.expand_dims(h_taps, -1) * gw + ops.expand_dims(w_taps, 1),
            (-1, 4),
        )
        corner_weights = ops.reshape(
            ops.expand_dims(h_weights, -1) * ops.expand_dims(w_weights, 1),
            (-1, 4),
        )
        return ops.sum(
            self.position_embedding_table(corner_indices)
            * ops.expand_dims(corner_weights, -1),
            axis=1,
        )

    def call(self, pixel_values, grid_thw):
        embeddings = self.patch_embedding(pixel_values)
        num_patches = ops.shape(pixel_values)[0]
        pos_embeds = self._bilinear_position_embeddings(grid_thw, num_patches)
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

    def _masked_full_attention(self, q, k, v, segment_id):
        """Every row attends over the whole sequence, masked to its own
        segment. `segment_id`: shape `(seq_len,)`.
        """
        q = ops.transpose(q, (1, 0, 2))
        k = ops.transpose(k, (1, 0, 2))
        v = ops.transpose(v, (1, 0, 2))

        scores = ops.matmul(q, ops.transpose(k, (0, 2, 1)))
        scores = ops.cast(scores * self._inv_scale, "float32")
        same_segment = ops.equal(
            ops.expand_dims(segment_id, 0), ops.expand_dims(segment_id, 1)
        )
        mask = ops.cast(same_segment, "float32")
        # Mask `scores` before `exp()`, not just the final probabilities,
        # to avoid overflow from out-of-segment raw scores.
        scores = ops.where(same_segment, scores, -1e9)
        row_max = ops.max(scores, axis=-1, keepdims=True)
        exp_scores = ops.exp(scores - row_max) * mask
        probs = exp_scores / ops.sum(exp_scores, axis=-1, keepdims=True)
        probs = ops.cast(probs, v.dtype)
        out = ops.matmul(probs, v)
        return ops.transpose(out, (1, 0, 2))

    def _padded_segment_attention(
        self, q, k, v, padded_index, num_slots, slot_size
    ):
        """Batched per-segment attention (a window, or a frame for full
        attention): scatter rows into a `(num_slots, slot_size)` buffer
        by segment and in-segment rank, matmul+softmax per slot, gather
        the real rows back out.
        """
        padded_len = num_slots * slot_size
        seq_len = ops.shape(q)[0]

        def scatter_pad(x):
            return ops.scatter(
                ops.expand_dims(padded_index, -1),
                x,
                (padded_len, self.num_heads, self.head_dim),
            )

        def to_blocks(x):
            x = ops.reshape(
                x, (num_slots, slot_size, self.num_heads, self.head_dim)
            )
            return ops.transpose(x, (0, 2, 1, 3))

        q_pad = to_blocks(scatter_pad(q))
        k_pad = to_blocks(scatter_pad(k))
        v_pad = to_blocks(scatter_pad(v))
        valid = ops.scatter(
            ops.expand_dims(padded_index, -1),
            ops.ones((seq_len,), dtype="float32"),
            (padded_len,),
        )
        valid_key = ops.reshape(valid, (num_slots, 1, 1, slot_size)) > 0

        scores = ops.matmul(q_pad, ops.transpose(k_pad, (0, 1, 3, 2)))
        scores = ops.cast(scores * self._inv_scale, "float32")
        scores = ops.where(valid_key, scores, -1e9)
        row_max = ops.max(scores, axis=-1, keepdims=True)
        exp_scores = ops.exp(scores - row_max) * ops.cast(valid_key, "float32")

        probs = exp_scores / ops.maximum(
            ops.sum(exp_scores, axis=-1, keepdims=True), 1e-9
        )
        probs = ops.cast(probs, v.dtype)
        out_pad = ops.matmul(probs, v_pad)
        out_pad = ops.transpose(out_pad, (0, 2, 1, 3))
        out_pad = ops.reshape(
            out_pad, (padded_len, self.num_heads, self.head_dim)
        )
        return ops.take(out_pad, padded_index, axis=0)

    def call(
        self,
        x,
        position_embeddings,
        segment_id,
        padded_index=None,
        num_slots=None,
        slot_size=None,
    ):
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

        if padded_index is not None:
            out = self._padded_segment_attention(
                q, k, v, padded_index, num_slots, slot_size
            )
        else:
            out = self._masked_full_attention(q, k, v, segment_id)

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

    def call(
        self,
        x,
        position_embeddings,
        segment_id,
        padded_index=None,
        num_slots=None,
        slot_size=None,
    ):
        x = x + self.attn(
            self.norm1(x),
            position_embeddings,
            segment_id,
            padded_index=padded_index,
            num_slots=num_slots,
            slot_size=slot_size,
        )
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


def _window_layout(grid_thw, num_patches, window_patches):
    """`window_index`/`reverse_indices`/`window_segment_id` grouping
    each image's (pre-merge) patch grid into `window_patches x
    window_patches` windows, for alternating window/full attention.
    """
    if len(ops.shape(grid_thw)) == 1:
        grid_thw = ops.expand_dims(grid_thw, axis=0)
    grid_thw = ops.cast(grid_thw, "int32")
    t, h, w = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
    m = window_patches

    lengths = t * h * w
    cumulative = ops.cumsum(lengths)
    starts = ops.concatenate(
        [ops.zeros((1,), dtype=cumulative.dtype), cumulative[:-1]]
    )
    num_window_cols = ops.floor_divide(w + m - 1, m)
    blocks_per_frame = ops.floor_divide(h + m - 1, m) * num_window_cols
    blocks_total = blocks_per_frame * t
    block_starts = ops.cumsum(blocks_total) - blocks_total

    positions = ops.arange(num_patches, dtype="int32")
    image_id = ops.cast(
        ops.searchsorted(cumulative, positions, side="right"), "int32"
    )
    local_offset = positions - ops.take(starts, image_id)

    h_p = ops.take(h, image_id)
    w_p = ops.take(w, image_id)
    hw_p = h_p * w_p
    offset_in_frame = ops.mod(local_offset, hw_p)
    frame_within_image = ops.floor_divide(local_offset, hw_p)

    row = ops.floor_divide(offset_in_frame, w_p)
    col = ops.mod(offset_in_frame, w_p)
    block_row = ops.floor_divide(row, m)
    block_col = ops.floor_divide(col, m)
    local_row = row - block_row * m
    local_col = col - block_col * m
    row_height = ops.minimum(m, h_p - block_row * m)
    col_width = ops.minimum(m, w_p - block_col * m)

    offset_within_frame_reordered = (
        block_row * m * w_p
        + row_height * block_col * m
        + local_row * col_width
        + local_col
    )
    reverse_indices = ops.cast(
        ops.take(starts, image_id)
        + frame_within_image * hw_p
        + offset_within_frame_reordered,
        "int32",
    )
    window_index = ops.scatter(
        ops.expand_dims(reverse_indices, -1), positions, (num_patches,)
    )

    orig_block_id = ops.cast(
        ops.take(block_starts, image_id)
        + frame_within_image * ops.take(blocks_per_frame, image_id)
        + block_row * ops.take(num_window_cols, image_id)
        + block_col,
        "int32",
    )
    window_segment_id = ops.take(orig_block_id, window_index)

    return window_index, reverse_indices, window_segment_id


def _shuffle_index_positions(grid_thw, num_patches, merge_size):
    """Pixel-shuffle merge permutation — the same block-transpose as
    `get_vision_pixel_shuffle_index`, as arithmetic instead of an actual
    reshape, so it traces without concrete `grid_thw` values.
    """
    row, col, _, w_per_patch, _, frame_start = _per_patch_coordinates(
        grid_thw, num_patches
    )
    offset_in_frame = row * w_per_patch + col
    block_size = merge_size * merge_size
    num_col_blocks = ops.floor_divide(w_per_patch, merge_size)
    stride = num_col_blocks * block_size

    block_row = ops.floor_divide(offset_in_frame, stride)
    remainder = ops.mod(offset_in_frame, stride)
    block_col = ops.floor_divide(remainder, block_size)
    remainder = ops.mod(remainder, block_size)
    sub_row = ops.floor_divide(remainder, merge_size)
    sub_col = ops.mod(remainder, merge_size)

    source_within_frame = (block_row * merge_size + sub_row) * w_per_patch + (
        block_col * merge_size + sub_col
    )
    return frame_start + source_within_frame


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
        max_num_windows: int or `None`. When set, window-attention layers
            batch each window into a fixed `(max_num_windows, window_
            patches ** 2)` buffer for one matmul+softmax per window slot,
            instead of one sequence-wide masked matmul — closer to plain
            per-window attention numerically. Must be at least the real
            number of windows in any call (across all images/frames);
            exceeding it produces incorrect (out-of-bounds) results.
            Defaults to `None` (the sequence-wide masked path, no cap).
        max_num_frames: int or `None`. Same idea as `max_num_windows`,
            for full-attention layers: batches each frame into a fixed
            `(max_num_frames, max_frame_size)` buffer. Requires
            `max_frame_size` too. Must be at least the real number of
            frames (across all images/videos) in any call.
        max_frame_size: int or `None`. The per-slot size for
            `max_num_frames` — must be at least the largest real
            per-frame patch count (`height * width`) in any call.
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
        max_num_windows=None,
        max_num_frames=None,
        max_frame_size=None,
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
        self.max_num_windows = max_num_windows
        self.max_num_frames = max_num_frames
        self.max_frame_size = max_frame_size
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

    def _rot_pos_emb(self, grid_thw, num_patches=None):
        if num_patches is None:
            grid_thw_np = ops.convert_to_numpy(grid_thw)
            if grid_thw_np.ndim == 1:
                grid_thw_np = grid_thw_np[np.newaxis, :]
            num_patches = int(np.sum(np.prod(grid_thw_np, axis=-1)))

        row, col, _, _, _, _ = _per_patch_coordinates(grid_thw, num_patches)
        positions = ops.stack([col + 1, row + 1], axis=-1)
        freqs = self.rotary_pos_emb.get_freqs(positions)
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

        hidden_states = self.patch_embedder(pixel_values, grid_thw)
        hidden_states = self.ln_pre(hidden_states)

        window_patches = max(self.window_size // self.patch_size, 1)
        window_index, reverse_indices, window_segment_id = _window_layout(
            grid_thw, num_patches, window_patches
        )
        hidden_states = ops.take(hidden_states, window_index, axis=0)

        window_padded_index, window_size_val = None, None
        if self.max_num_windows is not None:
            window_size_val = window_patches * window_patches
            positions = ops.arange(num_patches, dtype="int32")
            # `window_segment_id` is non-decreasing (windows are already
            # contiguous), so `searchsorted` against itself gives each
            # row's window's first position — hence its in-window rank.
            window_start = ops.cast(
                ops.searchsorted(
                    window_segment_id, window_segment_id, side="left"
                ),
                "int32",
            )
            local_rank = positions - window_start
            window_padded_index = (
                window_segment_id * window_size_val + local_rank
            )

        row, col, _, w_per_patch, full_segment_id, _ = _per_patch_coordinates(
            grid_thw, num_patches
        )
        frame_padded_index = None
        if self.max_num_frames is not None and self.max_frame_size is not None:
            offset_in_frame = row * w_per_patch + col
            frame_padded_index = (
                full_segment_id * self.max_frame_size + offset_in_frame
            )

        shuffle_index = _shuffle_index_positions(
            grid_thw, num_patches, self.merge_size
        )

        cos, sin = self._rot_pos_emb(grid_thw, num_patches)
        cos = ops.take(cos, window_index, axis=0)
        sin = ops.take(sin, window_index, axis=0)

        for layer, layer_type in zip(self.blocks, self.layer_types):
            if layer_type == "full_attention":
                segment_id = full_segment_id
                layer_padded_index = frame_padded_index
                num_slots, slot_size = self.max_num_frames, self.max_frame_size
            else:
                segment_id = window_segment_id
                layer_padded_index = window_padded_index
                num_slots, slot_size = self.max_num_windows, window_size_val
            if layer_padded_index is None:
                num_slots, slot_size = None, None
            hidden_states = layer(
                hidden_states,
                (cos, sin),
                segment_id,
                padded_index=layer_padded_index,
                num_slots=num_slots,
                slot_size=slot_size,
            )

        hidden_states = ops.take(hidden_states, reverse_indices, axis=0)
        hidden_states = self.ln_post(hidden_states)

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
                "max_num_windows": self.max_num_windows,
                "max_num_frames": self.max_num_frames,
                "max_frame_size": self.max_frame_size,
            }
        )
        return config
