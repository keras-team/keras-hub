import keras
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
            "fullatt_block_indexes": [1],
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
        # Non-default window_size/theta/epsilon must survive the
        # serialize/deserialize roundtrip.
        encoder = Qwen2_5VLVisionEncoder(
            **dict(
                self.init_kwargs,
                window_size=56,
                theta=50000.0,
                epsilon=1e-5,
            )
        )
        self.run_serialization_test(encoder)

    def test_fullatt_block_indexes_validation(self):
        # depth=2 in init_kwargs → valid indexes are 0 and 1 only.
        with self.assertRaises(ValueError):
            Qwen2_5VLVisionEncoder(
                **dict(self.init_kwargs, fullatt_block_indexes=[2])
            )
        with self.assertRaises(ValueError):
            Qwen2_5VLVisionEncoder(
                **dict(self.init_kwargs, fullatt_block_indexes=[-1])
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
        # Deterministic weights — init order varies across backends.
        rng = np.random.default_rng(0)
        encoder.build()
        for w in encoder.weights:
            w.assign(rng.normal(0.0, 0.05, size=w.shape).astype("float32"))

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
        # fp32 runs as tf32 on gpu; observed drift ~1.1e-3 on A100.
        tol = 5e-3 if running_on_gpu() else 1e-5
        self.assertAllClose(joint[:n_a], out_a, atol=tol, rtol=tol)
        self.assertAllClose(joint[n_a:], out_b, atol=tol, rtol=tol)

    def test_windowed_attention_matches_reference(self):
        """The block-sparse `window_ids` path must equal per-window
        softmax attention, including windows that straddle a
        `window_cap` block boundary (the 3*cap key gather)."""
        rng = np.random.default_rng(0)
        embed_dim, num_heads = 32, 4
        head_dim = embed_dim // num_heads
        cap = 16
        seq_len = 56
        x = rng.random((seq_len, embed_dim)).astype("float32")
        # Windows 1..3 each straddle a cap boundary: block 1 queries need
        # block 0 keys, etc.
        window_ids = np.array(
            [0] * 4 + [1] * 16 + [2] * 16 + [3] * 16 + [4] * 4,
            dtype="int32",
        )

        attn = Qwen2VLVisionAttention(embed_dim=embed_dim, num_heads=num_heads)
        # Identity RoPE (cos=1, sin=0) keeps the reference simple.
        cos = np.ones((seq_len, head_dim), dtype="float32")
        sin = np.zeros((seq_len, head_dim), dtype="float32")
        out = ops.convert_to_numpy(
            attn(x, (cos, sin), window_ids=window_ids, window_cap=cap)
        )

        # Numpy reference: fused qkv → per-window softmax → proj.
        qkv = x @ ops.convert_to_numpy(attn.qkv.kernel)
        qkv += ops.convert_to_numpy(attn.qkv.bias)
        qkv = qkv.reshape(seq_len, 3, num_heads, head_dim)
        q, k, v = (qkv[:, i].transpose(1, 0, 2) for i in range(3))
        sc = np.einsum("hid,hjd->hij", q, k) * head_dim**-0.5
        same = window_ids[:, None] == window_ids[None, :]
        sc = np.where(same[None], sc, -np.inf)
        sc = np.exp(sc - sc.max(axis=-1, keepdims=True))
        sc /= sc.sum(axis=-1, keepdims=True)
        ref = np.einsum("hij,hjd->hid", sc, v).transpose(1, 0, 2)
        ref = ref.reshape(seq_len, embed_dim)
        ref = ref @ ops.convert_to_numpy(attn.proj.kernel)
        ref += ops.convert_to_numpy(attn.proj.bias)

        tol = 1e-3 if running_on_gpu() else 1e-4
        self.assertAllClose(out, ref, atol=tol, rtol=tol)

    def test_windowed_attention_finite_gradients(self):
        """A `fit` step through windowed attention keeps weights finite
        when `seq_len % window_cap != 0` — padded query rows used to
        NaN-poison backward through softmax."""
        rng = np.random.default_rng(0)
        embed_dim, num_heads = 32, 4
        head_dim = embed_dim // num_heads
        cap = 16
        seq_len = 20  # 20 % 16 != 0 → 4 padded query rows.
        cos = np.ones((seq_len, head_dim), dtype="float32")
        sin = np.zeros((seq_len, head_dim), dtype="float32")
        window_ids = np.zeros(seq_len, dtype="int32")

        class AttentionFitModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.attn = Qwen2VLVisionAttention(
                    embed_dim=embed_dim, num_heads=num_heads
                )

            def call(self, x, training=False):
                # The vision tower is unbatched; unwrap the unit batch.
                # `x[0]` rather than squeeze: keras<=3.14 squeeze rejects
                # a dynamic (None) batch dim at trace time.
                out = self.attn(
                    x[0],
                    (cos, sin),
                    window_ids=window_ids,
                    window_cap=cap,
                )
                return ops.expand_dims(out, axis=0)

        model = AttentionFitModel()
        x = rng.random((1, seq_len, embed_dim)).astype("float32")
        y = np.zeros((1, seq_len, embed_dim), dtype="float32")
        model.compile(optimizer="sgd", loss="mse")
        model.fit(x, y, epochs=1, verbose=0)
        for w in model.trainable_weights:
            self.assertTrue(
                bool(np.all(np.isfinite(ops.convert_to_numpy(w)))),
                msg=f"non-finite weight after fit: {w.path}",
            )

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
        """`_window_layout` reproduces the reference window layout.

        Grid [1, 8, 10]: merge grid 4x5, window = 4 merge units per side
        → two windows: 4x4 block (16 units) + trailing column (4 units).
        """
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)
        window_index, unit_window = encoder._window_layout(
            np.array([[1, 8, 10]], dtype="int32"), num_units=20
        )
        # 4x4 raster block first, then the trailing column.
        block = [r * 5 + c for r in range(4) for c in range(4)]
        column = [r * 5 + 4 for r in range(4)]
        self.assertAllEqual(window_index, block + column)
        self.assertAllEqual(unit_window, ([0] * 4 + [1]) * 4)

        # Multi-row grid [[1, 12, 10]]: merge grid 6x5, ws=4 → windows at
        # wr=0 and wr=1, exercising multiple window rows.
        # Window 0 (r<4,c<4): units {0-3,5-8,10-13,15-18}, window 1
        # (r<4,c=4): {4,9,14,19}, window 2 (r>=4,c<4): {20-23,25-28},
        # window 3 (r>=4,c=4): {24,29}.
        window_index, unit_window = encoder._window_layout(
            np.array([[1, 12, 10]], dtype="int32"), num_units=30
        )
        w0 = [r * 5 + c for r in range(4) for c in range(4)]
        w1 = [r * 5 + 4 for r in range(4)]
        w2 = [r * 5 + c for r in range(4, 6) for c in range(4)]
        w3 = [r * 5 + 4 for r in range(4, 6)]
        self.assertAllEqual(window_index, w0 + w1 + w2 + w3)
        self.assertAllEqual(
            unit_window, ([0] * 4 + [1]) * 4 + ([2] * 4 + [3]) * 2
        )

    def test_window_unpermute_roundtrip(self):
        """argsort(window_index) restores the original merge-unit order,
        and permuted units sit in non-decreasing window order."""
        encoder = Qwen2_5VLVisionEncoder(**self.init_kwargs)
        # [[1,8,10],[2,6,8]] → merge grids 4x5 + 2x(3x4) = 20+24 = 44
        # units total.
        window_index, unit_window = encoder._window_layout(
            np.array([[1, 8, 10], [2, 6, 8]], dtype="int32"), num_units=44
        )
        units = ops.arange(44, dtype="int32")
        permuted = ops.take(units, window_index, axis=0)
        restored = ops.take(permuted, ops.argsort(window_index), axis=0)
        self.assertAllEqual(restored, units)

        # Windows must be contiguous in permuted order.
        permuted_windows = ops.convert_to_numpy(
            ops.take(unit_window, window_index, axis=0)
        )
        self.assertTrue(bool(np.all(np.diff(permuted_windows) >= 0)))

    def test_window_permutation_roundtrip_full_attention(self):
        """With only full-attention blocks, window permutation is a pure
        roundtrip: outputs do not depend on `window_size`."""
        rng = np.random.default_rng(0)
        kw = dict(self.init_kwargs, depth=2, fullatt_block_indexes=[0, 1])
        # Grid [1, 8, 16] → 128 patches. The default 112 window splits
        # the 4x8 merge grid in two; a 224 window holds it whole.
        patches = rng.random((128, 2, 14, 14, 3)).astype("float32")
        grid = np.array([[1, 8, 16]], dtype="int32")
        split = Qwen2_5VLVisionEncoder(**dict(kw, window_size=112))
        whole = Qwen2_5VLVisionEncoder(**dict(kw, window_size=224))
        split(patches, grid)
        whole(patches, grid)
        for w in split.weights:
            w.assign(rng.normal(0.0, 0.05, size=w.shape).astype("float32"))
        whole.set_weights(split.get_weights())

        out_split = ops.convert_to_numpy(split(patches, grid))
        out_whole = ops.convert_to_numpy(whole(patches, grid))
        tol = 2e-3 if running_on_gpu() else 1e-5
        self.assertAllClose(out_split, out_whole, atol=tol, rtol=tol)

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
        # Grid [1, 8, 16] → 128 patches (T is already temporal patches).
        patches = rng.random(
            (128, kw["temporal_patch_size"], 14, 14, 3)
        ).astype("float32")

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

        # Single-window grid [1, 8, 8] → 64 patches; both segment
        # choices are the same mask.
        grid_one_window = np.array([[1, 8, 8]], dtype="int32")
        out_w = ops.convert_to_numpy(windowed(patches[:64], grid_one_window))
        out_f = ops.convert_to_numpy(full(patches[:64], grid_one_window))
        tol = 2e-3 if running_on_gpu() else 1e-5
        self.assertAllClose(out_w, out_f, atol=tol, rtol=tol)
