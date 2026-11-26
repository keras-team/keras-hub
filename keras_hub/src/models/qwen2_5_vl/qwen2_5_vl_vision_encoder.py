import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLPatchMerger,
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

    norm_cls = QwenLayerNorm
    mlp_cls = Qwen2_5VLVisionMLP


class Qwen2_5VLPatchMerger(Qwen2VLPatchMerger):
    """Patch merger with RMSNorm ``ln_q`` (Qwen2.5-VL)."""

    norm_cls = QwenLayerNorm


@keras_hub_export("keras_hub.models.Qwen2_5VLVisionEncoder")
class Qwen2_5VLVisionEncoder(Qwen2VLVisionEncoder):
    """Vision encoder for Qwen2.5-VL (ViT with windowed attention).

    Unlike Qwen2-VL, which always attends within a single (image, temporal
    frame) segment, Qwen2.5-VL regroups merge units into
    ``window_size``-sized spatial windows
    (``window_size // spatial_merge_size // patch_size`` merge units per
    side) and runs windowed attention in most blocks. Blocks listed in
    ``fullatt_block_indexes`` keep per-(image, frame) attention. Tokens are
    permuted into window order before the blocks, and the merger output is
    unpermuted, so the returned tokens keep the original block-major
    layout.

    Args:
        patch_size: int. Spatial patch size in pixels. Defaults to ``14``.
        temporal_patch_size: int. Temporal patch size. Defaults to ``2``.
        in_channels: int. Input channels (typically 3). Defaults to ``3``.
        embed_dim: int. Vision hidden dimension. Defaults to ``1280``.
        out_dim: int. Output dimension (= text backbone ``hidden_dim``);
            ``None`` uses ``embed_dim``. Defaults to ``None``.
        depth: int. Number of ViT blocks. Defaults to ``32``.
        num_heads: int. Attention heads per ViT block. Defaults to ``16``.
        intermediate_dim: int. Vision MLP intermediate dimension;
            ``None`` uses ``3420``. Defaults to ``None``.
        spatial_merge_size: int. Spatial merge factor. Defaults to ``2``.
        window_size: int. Window attention size in pixels. Defaults to
            ``112``.
        fullatt_block_indexes: list of int. Block indexes that use
            per-(image, frame) attention instead of windowed attention.
            Defaults to ``(7, 15, 23, 31)``.
        theta: float. RoPE base frequency. Defaults to ``10000.0``.
        epsilon: float. RMSNorm epsilon. Defaults to ``1e-6``.
        dtype: compute dtype; the vision tower always runs in float32.
            Defaults to ``None``.

    Examples:

    ```python
    # Tiny encoder; one 4x4-patch image grid.
    encoder = keras_hub.models.Qwen2_5VLVisionEncoder(
        patch_size=2,
        temporal_patch_size=2,
        embed_dim=16,
        out_dim=8,
        depth=2,
        num_heads=4,
        intermediate_dim=32,
        window_size=4,
        fullatt_block_indexes=[1],
    )
    encoder.build()
    # (total_patches, T, pH, pW, C) patches + (num_images, 3) grid.
    pixel_values = np.random.rand(16, 2, 2, 2, 3).astype("float32")
    grid_thw = np.array([[1, 4, 4]], dtype="int32")
    merged = encoder(pixel_values, grid_thw)  # (4, 8) merged tokens
    ```
    """

    vision_block_cls = Qwen2_5VLVisionBlock
    patch_merger_cls = Qwen2_5VLPatchMerger

    def _vision_mlp_dim(self):
        return self.intermediate_dim

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
        if any(i < 0 or i >= depth for i in fullatt_block_indexes):
            raise ValueError(
                f"`fullatt_block_indexes` must lie in [0, depth); got "
                f"{list(fullatt_block_indexes)} with depth={depth}."
            )
        # `_vision_mlp_dim` reads this while the parent builds blocks.
        self.intermediate_dim = (
            intermediate_dim if intermediate_dim is not None else 3420
        )
        super().__init__(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            out_dim=out_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=self.intermediate_dim / embed_dim,
            spatial_merge_size=spatial_merge_size,
            theta=theta,
            epsilon=epsilon,
            dtype=dtype,
            **kwargs,
        )
        self.window_size = window_size
        self.fullatt_block_indexes = list(fullatt_block_indexes)

    def _window_layout(self, grid_thw, num_units):
        """Window reorder + per-unit window ids for windowed attention.

        Merge units are regrouped into ``ws x ws`` windows
        (``ws = window_size // spatial_merge_size // patch_size`` units
        per side) per (image, frame). Every output shape is static and
        only *values* depend on the grids — required for ``jax.jit`` /
        ``tf.function`` tracing.

        Args:
            grid_thw: int tensor ``(num_images, 3)`` with [T, H, W] per
                image (H, W in patches).
            num_units: int. Total merge-unit count
                (``sum(T * H/ms * W/ms)``).
        Returns:
            ``(window_index, unit_window)``:
            - ``window_index``: ``(num_units,)`` int32 gather order;
              ``take(units, window_index)`` groups each window's units
              contiguously.
            - ``unit_window``: ``(num_units,)`` int32 global window id of
              each unit *in original order*; permute via ``window_index``
              for per-token window ids.
        """
        ms = self.spatial_merge_size
        ws = self.window_size // ms // self.patch_size

        grid_thw = ops.cast(grid_thw, "int32")
        t = grid_thw[:, 0]
        lh = grid_thw[:, 1] // ms  # merge-grid dims
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
        lh_i = ops.take(lh, img_idx, axis=0)
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

        # Real units fill an ``rh x cw`` top-left rectangle in raster
        # order, so rank = ``ir * cw + ic`` at the window's cumsum start.
        ir = r - wr * ws
        ic = c - wc * ws
        cw_i = ops.minimum(ws, lw_i - wc * ws)
        unit_rank = ir * cw_i + ic

        rh_i = ops.minimum(ws, lh_i - wr * ws)
        win_units = rh_i * cw_i
        counts = ops.scatter_update(
            ops.zeros((num_units,), "int32"),
            ops.expand_dims(unit_window, axis=1),
            win_units,
        )
        emit_starts = ops.cumsum(counts) - counts
        emit = ops.take(emit_starts, unit_window, axis=0) + unit_rank
        window_index = ops.argsort(emit)
        return (
            ops.cast(window_index, "int32"),
            ops.cast(unit_window, "int32"),
        )

    def _encode_tokens(self, hidden_states, grid_thw):
        seq_len = ops.shape(hidden_states)[0]
        cos_emb, sin_emb = self._rot_pos_emb(grid_thw, seq_len)
        frame_segs = self._segment_ids(grid_thw, seq_len)
        ms2 = self.spatial_merge_size * self.spatial_merge_size

        # Regroup merge units window-major and permute tokens, rotary
        # embeddings and segment ids together.
        num_units = seq_len // ms2
        window_index, unit_window = self._window_layout(grid_thw, num_units)

        def _permute(x, dim):
            x = ops.reshape(x, (num_units, ms2, dim))
            x = ops.take(x, window_index, axis=0)
            return ops.reshape(x, (seq_len, dim))

        hidden_states = _permute(hidden_states, self.embed_dim)
        cos_emb = _permute(cos_emb, self.head_dim)
        sin_emb = _permute(sin_emb, self.head_dim)

        # Per-token window/frame ids in permuted order, taken per merge
        # unit and repeated over its ms2 tokens.
        unit_frame_segs = ops.take(
            ops.reshape(frame_segs, (num_units, ms2)), 0, axis=1
        )
        window_ids = ops.repeat(
            ops.take(unit_window, window_index, axis=0), ms2
        )
        frame_segs = ops.repeat(
            ops.take(unit_frame_segs, window_index, axis=0), ms2
        )
        ws = self.window_size // self.spatial_merge_size // self.patch_size
        window_cap = ws * ws * ms2

        fullatt = set(self.fullatt_block_indexes)
        position_embeddings = (cos_emb, sin_emb)
        for i, blk in enumerate(self.blocks):
            if i in fullatt:
                hidden_states = blk(
                    hidden_states, position_embeddings, frame_segs
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    position_embeddings,
                    window_ids=window_ids,
                    window_cap=window_cap,
                )

        merged = self.merger(hidden_states)  # (num_units, out_dim)
        # Undo the window permutation; the base `call` then drops the
        # appended dummy unit.
        return ops.take(merged, ops.argsort(window_index), axis=0)

    def get_config(self):
        config = super().get_config()
        # `intermediate_dim` parameterizes the MLP, not `mlp_ratio`.
        config.pop("mlp_ratio")
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "window_size": self.window_size,
                "fullatt_block_indexes": self.fullatt_block_indexes,
            }
        )
        return config
