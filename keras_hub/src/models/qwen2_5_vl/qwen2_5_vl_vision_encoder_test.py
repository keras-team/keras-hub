import numpy as np
from keras import ops

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_vision_encoder import (
    Qwen2_5VLVisionEncoder,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_vision_encoder import (
    Qwen2VLVisionAttention,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu


class Qwen2_5VLVisionEncoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "embed_dim": 64,
            "out_dim": 128,
            "depth": 2,
            "num_heads": 4,
            "intermediate_dim": 256,
            "spatial_merge_size": 2,
        }

    def _patches(self, n_patches, seed=0):
        rng = np.random.default_rng(seed)
        kw = self.init_kwargs
        return rng.random(
            (
                n_patches,
                kw["temporal_patch_size"],
                kw["patch_size"],
                kw["patch_size"],
                kw["in_channels"],
            )
        ).astype("float32")

    def test_vision_encoder_basics(self):
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)

        # 1 image with t=2, h=2, w=2 → total_patches = 8
        grid_thw = np.array([[2, 2, 2]], dtype="int32")
        total_patches = int(np.prod(grid_thw))
        hidden_states = self._patches(total_patches)

        encoder(hidden_states, grid_thw)  # build
        # Deterministic weights so the value checks below run on every
        # backend. QwenLayerNorm uses a `1.0 + scale` gain, so zero-centered
        # noise keeps all norms ~identity.
        rng = np.random.default_rng(0)
        for w in encoder.weights:
            values = rng.normal(0.0, 0.05, size=w.shape)
            w.assign(values.astype("float32"))

        output = encoder(hidden_states, grid_thw)

        # After merger, should reduce by spatial_merge_size^2
        merge_sq = self.init_kwargs["spatial_merge_size"] ** 2
        expected_tokens = total_patches // merge_sq
        self.assertEqual(
            tuple(output.shape), (expected_tokens, self.init_kwargs["out_dim"])
        )

        output = ops.convert_to_numpy(output)
        # fp32 runs as tf32 on gpu and drifts to ~1e-3 over the full
        # encoder; keep 2x headroom over observed diffs to avoid flakes.
        tol = 2e-3 if running_on_gpu() else 1e-4
        np.testing.assert_allclose(
            output[0, :6],
            [-0.010924, -0.028941, -0.076252, -0.052837, -0.00358, 0.05001],
            atol=tol,
            rtol=tol,
        )
        np.testing.assert_allclose(
            output[1, :4],
            [-0.02005, -0.021227, -0.074488, -0.03363],
            atol=tol,
            rtol=tol,
        )

    def test_vision_encoder_config_roundtrip(self):
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)
        config = encoder.get_config()
        new_encoder = Qwen2_5VLVisionEncoder.from_config(config)

        # Verify config values match
        self.assertEqual(encoder.patch_size, new_encoder.patch_size)
        self.assertEqual(
            encoder.temporal_patch_size, new_encoder.temporal_patch_size
        )
        self.assertEqual(encoder.in_channels, new_encoder.in_channels)
        self.assertEqual(encoder.embed_dim, new_encoder.embed_dim)
        self.assertEqual(encoder.out_dim, new_encoder.out_dim)
        self.assertEqual(encoder.depth, new_encoder.depth)
        self.assertEqual(encoder.num_heads, new_encoder.num_heads)
        self.assertEqual(encoder.intermediate_dim, new_encoder.intermediate_dim)
        self.assertEqual(
            encoder.spatial_merge_size, new_encoder.spatial_merge_size
        )
        self.assertEqual(encoder.window_size, new_encoder.window_size)
        self.assertEqual(
            encoder.fullatt_block_indexes, new_encoder.fullatt_block_indexes
        )

    def test_vision_encoder_with_multiple_images(self):
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)

        # 2 images with different grid sizes
        grid_thw = np.array([[2, 2, 2], [2, 4, 4]], dtype="int32")
        total_patches = int(np.sum(np.prod(grid_thw, axis=1)))
        hidden_states = self._patches(total_patches)

        output = encoder(hidden_states, grid_thw)

        merge_sq = self.init_kwargs["spatial_merge_size"] ** 2
        expected_tokens = total_patches // merge_sq
        self.assertEqual(
            tuple(output.shape), (expected_tokens, self.init_kwargs["out_dim"])
        )

    def test_per_frame_attention_isolation(self):
        """Attention is confined per (image, frame): concatenating two
        images must produce the same per-image outputs as processing them
        separately."""
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)

        grid_a = np.array([[1, 2, 2]], dtype="int32")
        grid_b = np.array([[1, 4, 4]], dtype="int32")
        patches_a = self._patches(int(np.prod(grid_a)), seed=1)
        patches_b = self._patches(int(np.prod(grid_b)), seed=2)

        out_a = ops.convert_to_numpy(encoder(patches_a, grid_a))
        out_b = ops.convert_to_numpy(encoder(patches_b, grid_b))

        joint = encoder(
            np.concatenate([patches_a, patches_b], axis=0),
            np.concatenate([grid_a, grid_b], axis=0),
        )
        joint = ops.convert_to_numpy(joint)

        n_a = out_a.shape[0]
        tol = 2e-3 if running_on_gpu() else 1e-5
        np.testing.assert_allclose(joint[:n_a], out_a, atol=tol, rtol=tol)
        np.testing.assert_allclose(joint[n_a:], out_b, atol=tol, rtol=tol)

    def test_chunked_attention_matches_reference(self):
        """seq_len > 1024 takes the chunked path; its output must equal a
        plain per-segment softmax attention computed in numpy."""
        rng = np.random.default_rng(0)
        embed_dim, num_heads = 32, 4
        head_dim = embed_dim // num_heads
        seq_len = 1030  # > chunk → also exercises the clamped tail chunk.
        x = rng.random((seq_len, embed_dim)).astype("float32")
        # Two segments, so the mask is exercised across chunk boundaries.
        seg = np.repeat([0, 1], [600, seq_len - 600]).astype("int32")

        attn = Qwen2VLVisionAttention(embed_dim=embed_dim, num_heads=num_heads)
        # Identity RoPE (cos=1, sin=0) keeps the reference simple.
        cos = np.ones((seq_len, head_dim), dtype="float32")
        sin = np.zeros((seq_len, head_dim), dtype="float32")
        out = ops.convert_to_numpy(attn(x, (cos, sin), seg))

        # Numpy reference: fused qkv → per-segment softmax → proj.
        qkv = x @ ops.convert_to_numpy(attn.qkv.kernel)
        qkv += ops.convert_to_numpy(attn.qkv.bias)
        qkv = qkv.reshape(seq_len, 3, num_heads, head_dim)
        q, k, v = (qkv[:, i].transpose(1, 0, 2) for i in range(3))
        sc = np.einsum("hid,hjd->hij", q, k) * head_dim**-0.5
        same = seg[:, None] == seg[None, :]
        sc = np.where(same[None], sc, -np.inf)
        sc = np.exp(sc - sc.max(axis=-1, keepdims=True))
        sc /= sc.sum(axis=-1, keepdims=True)
        ref = np.einsum("hij,hjd->hid", sc, v).transpose(1, 0, 2)
        ref = ref.reshape(seq_len, embed_dim)
        ref = ref @ ops.convert_to_numpy(attn.proj.kernel)
        ref += ops.convert_to_numpy(attn.proj.bias)

        tol = 1e-3 if running_on_gpu() else 1e-4
        np.testing.assert_allclose(out, ref, atol=tol, rtol=tol)

    def test_empty_input(self):
        """Zero-patch input returns an empty output (text-only path)."""
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)
        kw = self.init_kwargs
        patches = np.zeros(
            (
                0,
                kw["temporal_patch_size"],
                kw["patch_size"],
                kw["patch_size"],
                kw["in_channels"],
            ),
            dtype="float32",
        )
        grid_thw = np.zeros((0, 3), dtype="int32")
        output = encoder(patches, grid_thw)
        self.assertEqual(tuple(output.shape), (0, kw["out_dim"]))

    def test_rotary_embeddings(self):
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)

        # grid [1, 2, 2] → 4 patches, head_dim = 64/4 = 16.
        grid_thw = np.array([[1, 2, 2]], dtype="int32")
        cos, sin = encoder._rot_pos_emb(grid_thw, seq_len=4)

        self.assertEqual(tuple(cos.shape), (4, 16))
        self.assertEqual(tuple(sin.shape), (4, 16))

    def test_window_layout_matches_hf(self):
        """`_window_layout` reproduces HF `get_vision_window_index`.

        Grid [1, 8, 10]: merge grid 4x5, window = 4 merge units per side
        → two windows: 4x4 block (16 units) + trailing column (4 units).
        """
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)
        window_index, unit_window = encoder._window_layout(
            np.array([[1, 8, 10]], dtype="int32"), num_units=20
        )
        self.assertAllEqual(
            window_index,
            [
                0,
                1,
                2,
                3,
                5,
                6,
                7,
                8,
                10,
                11,
                12,
                13,
                15,
                16,
                17,
                18,
                4,
                9,
                14,
                19,
            ],
        )
        self.assertAllEqual(
            unit_window,
            [0] * 4 + [1] + [0] * 4 + [1] + [0] * 4 + [1] + [0] * 4 + [1],
        )

    def test_window_unpermute_roundtrip(self):
        """argsort(window_index) restores the original merge-unit order."""
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)
        window_index, _ = encoder._window_layout(
            np.array([[1, 8, 10], [2, 6, 8]], dtype="int32"), num_units=32
        )
        units = ops.arange(32, dtype="int32")
        permuted = ops.take(units, window_index, axis=0)
        restored = ops.take(permuted, ops.argsort(window_index), axis=0)
        self.assertAllEqual(restored, units)

    def test_window_vs_full_attention(self):
        """Windowed and full-attention blocks produce different segment
        masks on multi-window grids, and identical masks when the image
        fits in a single window.

        Grid [1, 8, 16]: merge grid 4x8, window 4x4 → 2 windows per frame.
        With every block in `fullatt_block_indexes`, tokens attend across
        the whole frame; windowed blocks restrict to their window.
        """
        rng = np.random.default_rng(0)
        kw = dict(self.init_kwargs, depth=1)
        patches = rng.random((64, kw["temporal_patch_size"], 14, 14, 3)).astype(
            "float32"
        )

        grid_two_windows = np.array([[1, 8, 16]], dtype="int32")
        windowed = Qwen2_5VLVisionEncoder(**dict(kw, fullatt_block_indexes=[]))
        full = Qwen2_5VLVisionEncoder(**dict(kw, fullatt_block_indexes=[0]))
        # Same weights → differences come only from the attention mask.
        windowed(patches, grid_two_windows)
        full(patches, grid_two_windows)
        full.set_weights(windowed.get_weights())
        out_w = ops.convert_to_numpy(windowed(patches, grid_two_windows))
        out_f = ops.convert_to_numpy(full(patches, grid_two_windows))
        self.assertGreater(float(np.max(np.abs(out_w - out_f))), 1e-3)

        # Single-window grid: both segment choices are the same mask.
        grid_one_window = np.array([[1, 8, 8]], dtype="int32")
        out_w = ops.convert_to_numpy(windowed(patches[:32], grid_one_window))
        out_f = ops.convert_to_numpy(full(patches[:32], grid_one_window))
        tol = 2e-3 if running_on_gpu() else 1e-5
        np.testing.assert_allclose(out_w, out_f, atol=tol, rtol=tol)
