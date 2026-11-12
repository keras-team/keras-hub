import math

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLPatchMerger,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionAttention,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionBlock,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionEncoder,
)


class Qwen2_5VLVisionMLP(keras.layers.Layer):
    """Vision MLP: gated SiLU ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Qwen2.5-VL replaces Qwen2-VL's ``fc1 -> QuickGELU -> fc2`` MLP with the
    same gated projection used by the text decoder, but with bias.
    """

    def __init__(self, embed_dim, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.gate_proj = keras.layers.Dense(
            self.mlp_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="gate_proj",
        )
        self.gate_proj.build(input_shape)
        self.up_proj = keras.layers.Dense(
            self.mlp_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.up_proj.build(input_shape)
        self.down_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="down_proj",
        )
        self.down_proj.build((input_shape[0], self.mlp_dim))
        self.built = True

    def call(self, x):
        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim})
        return config


class Qwen2_5VLVisionBlock(Qwen2VLVisionBlock):
    """Single vision transformer block: RMSNorm -> Attn -> RMSNorm -> MLP.

    Qwen2.5-VL swaps Qwen2-VL's ``LayerNormalization`` for RMSNorm and its
    QuickGELU MLP for a gated SiLU MLP. The residual structure and the
    attention submodule are unchanged.
    """

    def build(self, input_shape):
        self.norm1 = QwenLayerNorm(
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
        self.norm2 = QwenLayerNorm(
            epsilon=self.epsilon, dtype="float32", name="norm2"
        )
        self.norm2.build(input_shape)
        self.mlp = Qwen2_5VLVisionMLP(
            embed_dim=self.embed_dim,
            mlp_dim=self.mlp_dim,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(input_shape)
        self.built = True


class Qwen2_5VLPatchMerger(Qwen2VLPatchMerger):
    """Patch merger with RMSNorm ``ln_q`` (Qwen2.5-VL)."""

    def build(self, input_shape):
        self.ln_q = QwenLayerNorm(
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


@keras_hub_export("keras_hub.models.Qwen2_5VLVisionEncoder")
class Qwen2_5VLVisionEncoder(Qwen2VLVisionEncoder):
    """Vision encoder for Qwen2.5-VL (ViT with windowed attention).

    Unlike Qwen2-VL, which always attends within a single (image, temporal
    frame) segment, Qwen2.5-VL regroups merge units into
    ``window_size``-sized spatial windows (``vit_merger_window_size =
    window_size // spatial_merge_size // patch_size`` units per side) and
    runs windowed attention in most blocks. Blocks listed in
    ``fullatt_block_indexes`` keep per-(image, frame) attention. Tokens are
    permuted into window order before the blocks (matching HF
    ``get_vision_window_index``) and the merger output is unpermuted, so
    the returned tokens keep the original block-major layout.

    Args:
        patch_size: int. Spatial patch size in pixels.
        temporal_patch_size: int. Temporal patch size.
        in_channels: int. Input channels (typically 3).
        embed_dim: int. Vision hidden dimension.
        out_dim: int. Output dimension (= text backbone ``hidden_dim``).
        depth: int. Number of ViT blocks.
        num_heads: int. Attention heads per ViT block.
        intermediate_dim: int. Vision MLP intermediate dimension
            (HF ``vision_config.intermediate_size``). Defaults to
            ``embed_dim * 4``.
        spatial_merge_size: int. Spatial merge factor.
        window_size: int. Window attention size in pixels.
        fullatt_block_indexes: list of int. Block indexes that use
            per-(image, frame) attention instead of windowed attention.
        theta: float. RoPE base frequency.
        epsilon: float. RMSNorm epsilon.
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
        intermediate_dim=None,
        spatial_merge_size=2,
        window_size=112,
        fullatt_block_indexes=(7, 15, 23, 31),
        theta=10000.0,
        epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        # Always run the vision encoder in float32 for numerical stability.
        keras.Model.__init__(self, dtype="float32", **kwargs)

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_dim = out_dim if out_dim is not None else embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.intermediate_dim = (
            intermediate_dim if intermediate_dim is not None else embed_dim * 4
        )
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.fullatt_block_indexes = list(fullatt_block_indexes)
        self.theta = theta
        self.epsilon = epsilon

        self.head_dim = embed_dim // num_heads

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
            Qwen2_5VLVisionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=self.intermediate_dim,
                epsilon=epsilon,
                dtype=self.dtype_policy,
                name=f"blocks_{i}",
            )
            for i in range(depth)
        ]

        self.merger = Qwen2_5VLPatchMerger(
            embed_dim=embed_dim,
            out_dim=self.out_dim,
            spatial_merge_size=spatial_merge_size,
            epsilon=epsilon,
            dtype=self.dtype_policy,
            name="merger",
        )

    def _window_layout(self, grid_thw, num_units):
        """Window reorder + per-unit window ids for windowed attention.

        Vectorized port of HF ``get_vision_window_index``: merge units are
        regrouped into ``ws x ws`` windows
        (``ws = window_size // spatial_merge_size // patch_size`` units per
        side) per (image, frame). Every output shape is static and only
        *values* depend on the grids — required for ``jax.jit`` /
        ``tf.function`` compilation, where grid values are dynamic.

        Real cells inside a window always form a ``rh x cw`` top-left
        rectangle, so a unit's rank inside its window is ``ir * cw + ic``
        and its emission position is its window's exclusive-cumsum start
        plus that rank.

        Args:
            grid_thw: int tensor ``(num_images, 3)`` with [T, H, W] per
                image (H, W in patches).
            num_units: int. Total merge-unit count
                (``sum(T * H/ms * W/ms)``).
        Returns:
            ``(window_index, unit_window)``:
            - ``window_index``: ``(num_units,)`` int32 gather order;
              ``take(units, window_index)`` groups each window's units
              contiguously (HF ``window_index`` semantics).
            - ``unit_window``: ``(num_units,)`` int32 global window id of
              each unit *in original order*; permute via ``window_index``
              for per-token segment ids.
        """
        ms = self.spatial_merge_size
        ws = self.window_size // ms // self.patch_size

        grid_thw = ops.cast(grid_thw, "int32")
        t = grid_thw[:, 0]
        lh = grid_thw[:, 1] // ms  # merge-grid dims (HF `llm_grid`)
        lw = grid_thw[:, 2] // ms
        units_per_img = t * lh * lw

        # Image index per unit via cumulative end offsets.
        ends = ops.cumsum(units_per_img)
        u = ops.arange(num_units, dtype="int32")
        img_idx = ops.sum(
            ops.cast(
                ops.expand_dims(u, 1) >= ops.expand_dims(ends, 0), "int32"
            ),
            axis=1,
        )
        img_idx = ops.minimum(img_idx, ops.shape(grid_thw)[0] - 1)
        starts = ends - units_per_img
        within = u - ops.take(starts, img_idx, axis=0)

        hw_i = ops.take(lh * lw, img_idx, axis=0)
        lw_i = ops.take(lw, img_idx, axis=0)
        frame = within // hw_i
        j = within - frame * hw_i
        r = j // lw_i
        c = j - r * lw_i

        # Window coords within each frame.
        wr = r // ws
        wc = c // ws

        # Global window id in emission order (image, frame, wr, wc).
        nw_h = (lh + ws - 1) // ws  # per-image window counts
        nw_w = (lw + ws - 1) // ws
        win_per_img = t * nw_h * nw_w
        win_off = ops.cumsum(win_per_img) - win_per_img
        nw_h_i = ops.take(nw_h, img_idx, axis=0)
        nw_w_i = ops.take(nw_w, img_idx, axis=0)
        local_win = (frame * nw_h_i + wr) * nw_w_i + wc
        unit_window = ops.take(win_off, img_idx, axis=0) + local_win

        # `window_index` is a stable sort of units by window id: inside a
        # window HF emits units in (ir, ic) row-major order, which
        # coincides with their original order — so the sort only needs
        # stability within equal window ids, encoded explicitly in the
        # key (sort stability is not guaranteed cross-backend).
        key = ops.cast(unit_window, "int64") * ops.cast(
            num_units, "int64"
        ) + ops.cast(u, "int64")
        window_index = ops.cast(ops.argsort(key), "int32")
        return window_index, ops.cast(unit_window, "int32")

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

        # Text-only calls feed a zero-sized pixel_values tensor. When the
        # patch count is statically zero, skip the transformer entirely.
        if pixel_values.shape[0] == 0:
            merged = ops.zeros((0, self.out_dim), dtype="float32")
            if batched:
                merged = ops.expand_dims(merged, axis=0)
            return merged

        # `ops.cond` can't express a 0-vs-N patch branch under `jax.jit`
        # (branches must return equal shapes). Instead, always encode by
        # appending one dummy merge group and gathering the real tokens
        # back out.
        num_patches = ops.shape(pixel_values)[0]
        ms2 = self.spatial_merge_size * self.spatial_merge_size
        num_merged = num_patches // ms2
        pad_shape = (ms2,) + (tuple(ops.shape(pixel_values)[1:]))
        pixel_values = ops.concatenate(
            [
                pixel_values,
                ops.zeros(pad_shape, dtype=pixel_values.dtype),
            ],
            axis=0,
        )
        pad_grid = ops.cast(
            ops.array([[1, self.spatial_merge_size, self.spatial_merge_size]]),
            grid_thw.dtype,
        )
        grid_thw = ops.concatenate([grid_thw, pad_grid], axis=0)

        # Patch embedding: each patch is one Conv3D "sample".
        hidden_states = self.patch_embed(pixel_values)  # (N,1,1,1,E)
        hidden_states = ops.reshape(
            hidden_states, (ops.shape(hidden_states)[0], self.embed_dim)
        )
        hidden_states = ops.cast(hidden_states, "float32")

        seq_len = ops.shape(hidden_states)[0]
        cos_emb, sin_emb = self._rot_pos_emb(grid_thw, seq_len)
        frame_segs = self._segment_ids(grid_thw, seq_len)

        # Window attention: regroup merge units window-major and permute
        # tokens, rotary embeddings and segment ids together (HF
        # `get_vision_window_index` + `permute`).
        num_units = seq_len // ms2
        window_index, unit_window = self._window_layout(grid_thw, num_units)

        def _permute(x, dim):
            x = ops.reshape(x, (num_units, ms2, dim))
            x = ops.take(x, window_index, axis=0)
            return ops.reshape(x, (seq_len, dim))

        hidden_states = _permute(hidden_states, self.embed_dim)
        cos_emb = _permute(cos_emb, self.head_dim)
        sin_emb = _permute(sin_emb, self.head_dim)

        # Per-token segment ids in permuted order. Each merge unit's ms2
        # tokens share one segment (merging never crosses frames/windows),
        # so ids are taken per-unit then repeated.
        unit_frame_segs = ops.take(
            ops.reshape(frame_segs, (num_units, ms2)), 0, axis=1
        )
        window_segs = ops.repeat(
            ops.take(unit_window, window_index, axis=0), ms2
        )
        frame_segs = ops.repeat(
            ops.take(unit_frame_segs, window_index, axis=0), ms2
        )

        fullatt = set(self.fullatt_block_indexes)
        position_embeddings = (cos_emb, sin_emb)
        for i, blk in enumerate(self.blocks):
            segment_ids = frame_segs if i in fullatt else window_segs
            hidden_states = blk(hidden_states, position_embeddings, segment_ids)

        merged = self.merger(hidden_states)  # (num_units, out_dim)
        # Undo the window permutation (HF `reverse_indices`), then drop
        # the appended dummy unit.
        merged = ops.take(merged, ops.argsort(window_index), axis=0)
        merged = ops.take(merged, ops.arange(num_merged), axis=0)

        if batched:
            merged = ops.expand_dims(merged, axis=0)
        return merged

    def get_config(self):
        # Skip `Qwen2VLVisionEncoder.get_config`: its fields (`mlp_ratio`)
        # do not exist on this class — 2.5-VL uses `intermediate_dim` and a
        # gated MLP instead. Base-layer config still carries
        # name/trainable/dtype.
        config = keras.layers.Layer.get_config(self)
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "embed_dim": self.embed_dim,
                "out_dim": self.out_dim,
                "depth": self.depth,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "spatial_merge_size": self.spatial_merge_size,
                "window_size": self.window_size,
                "fullatt_block_indexes": self.fullatt_block_indexes,
                "theta": self.theta,
                "epsilon": self.epsilon,
            }
        )
        return config
