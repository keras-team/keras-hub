"""Parity tests for the litertlm "traceable op" patches in ``traceable_ops.py``.

``traceable_ops.py`` temporarily replaces a handful of Keras torch-backend
ops (``one_hot``, ``repeat``, ``slice``, ``take``, ``scatter_update``,
``dot_product_attention``, ``arange``) with reimplementations that are
friendlier to ``torch.export`` tracing (see each function's docstring for
the specific tracing issue it works around). These tests do not exercise
``torch.export`` at all -- they call the patched functions directly, on
ordinary eager tensors, and check that they compute the exact same values
as the original (unpatched) Keras implementation on representative
shapes/dtypes/axes so a future change to one of these patches cannot
silently change numerics without a failing test.
"""

import unittest

import keras
import numpy as np
import torch

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm import traceable_ops


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@unittest.skipUnless(
    keras.config.backend() == "torch",
    "The litertlm traceable-op patches only exist for the PyTorch backend.",
)
class TraceableOpsParityTest(TestCase):
    # ------------------------------------------------------------------
    # one_hot
    # ------------------------------------------------------------------
    def test_one_hot_matches_original(self):
        from keras.src.backend.torch import nn as torch_backend_nn

        cases = [
            ([0, 1, 2, 3], 5, -1, "float32"),
            ([0, 1, 2, 3], 5, 0, "float32"),
            ([[0, 1], [2, 3]], 4, -1, "float32"),
            ([[0, 1], [2, 3]], 4, 1, "float32"),
            ([[0, 1], [2, 3]], 4, 2, "int32"),
            ([-1, 0, 2], 4, -1, "float32"),  # negative-index handling
        ]
        for x, num_classes, axis, dtype in cases:
            with self.subTest(x=x, num_classes=num_classes, axis=axis):
                x_t = torch.tensor(x, dtype=torch.int32)
                original = torch_backend_nn.one_hot(
                    x_t, num_classes, axis=axis, dtype=dtype
                )
                patched = traceable_ops._patched_one_hot(
                    x_t, num_classes, axis=axis, dtype=dtype
                )
                self.assertEqual(tuple(original.shape), tuple(patched.shape))
                self.assertAllClose(_to_np(original), _to_np(patched))

    def test_one_hot_rejects_sparse(self):
        from keras.src.backend.torch import nn as torch_backend_nn

        with self.assertRaises(ValueError):
            torch_backend_nn.one_hot(torch.tensor([0]), 4, sparse=True)
        with self.assertRaises(ValueError):
            traceable_ops._patched_one_hot(torch.tensor([0]), 4, sparse=True)

    # ------------------------------------------------------------------
    # repeat
    # ------------------------------------------------------------------
    def test_repeat_matches_original(self):
        from keras.src.backend.torch import numpy as torch_backend_numpy

        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        for axis in (0, 1, 2, -1):
            for repeats in (1, 2, 3):
                with self.subTest(axis=axis, repeats=repeats):
                    original = torch_backend_numpy.repeat(x, repeats, axis=axis)
                    patched = traceable_ops._traceable_repeat(
                        x, repeats, axis=axis
                    )
                    self.assertAllClose(_to_np(original), _to_np(patched))

    def test_repeat_scalar_tensor_repeats_matches_original(self):
        # A 0-D tensor `repeats` is not a plain Python `int`, so the
        # upstream Keras `repeat` falls through to the `repeat_interleave`
        # path, while `_traceable_repeat`'s `_is_scalar_integer` check takes
        # the unsqueeze+expand+reshape fast path instead. Values must match.
        from keras.src.backend.torch import numpy as torch_backend_numpy

        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        repeats = torch.tensor(2)
        original = torch_backend_numpy.repeat(x, repeats, axis=1)
        patched = traceable_ops._traceable_repeat(x, repeats, axis=1)
        self.assertAllClose(_to_np(original), _to_np(patched))

    def test_repeat_identity_when_repeats_is_one(self):
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        patched = traceable_ops._traceable_repeat(x, 1, axis=0)
        self.assertAllClose(_to_np(x), _to_np(patched))

    def test_repeat_rejects_negative(self):
        with self.assertRaises(ValueError):
            traceable_ops._traceable_repeat(torch.zeros(2, 2), -1, axis=0)

    def test_repeat_no_axis_falls_back_to_original(self):
        from keras.src.backend.torch import numpy as torch_backend_numpy

        x = torch.arange(6, dtype=torch.float32)
        original = torch_backend_numpy.repeat(x, 3, axis=None)
        patched = traceable_ops._traceable_repeat(x, 3, axis=None)
        self.assertAllClose(_to_np(original), _to_np(patched))

    # ------------------------------------------------------------------
    # slice (via `_make_patched_slice`)
    # ------------------------------------------------------------------
    def test_slice_static_matches_original(self):
        from keras.src.backend.torch import core as torch_core

        x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        patched_slice = traceable_ops._make_patched_slice()
        original = torch_core.slice(x, [1, 1, 0], [2, 2, 5])
        patched = patched_slice(x, [1, 1, 0], [2, 2, 5])
        self.assertAllClose(_to_np(original), _to_np(patched))

    def test_slice_single_dynamic_dim_matches_original(self):
        from keras.src.backend.torch import core as torch_core

        x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        patched_slice = traceable_ops._make_patched_slice()
        for start in range(0, 3):
            with self.subTest(start=start):
                start_t = torch.tensor(start)
                original = torch_core.slice(x, [start, 0, 0], [1, 4, 5])
                patched = patched_slice(x, [start_t, 0, 0], [1, 4, 5])
                self.assertAllClose(_to_np(original), _to_np(patched))

    def test_slice_dynamic_middle_dim_matches_original(self):
        # Regression check for the KV-cache read pattern used by
        # `KerasHubLiteRTAdapter`: slicing along a non-leading axis with a
        # dynamic start while the surrounding axes are fully covered.
        from keras.src.backend.torch import core as torch_core

        x = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
        patched_slice = traceable_ops._make_patched_slice()
        for start in range(0, 3):
            with self.subTest(start=start):
                start_t = torch.tensor(start)
                original = torch_core.slice(x, [0, start, 0], [2, 1, 5])
                patched = patched_slice(x, [0, start_t, 0], [2, 1, 5])
                self.assertAllClose(_to_np(original), _to_np(patched))

    def test_slice_multiple_dynamic_dims_raises(self):
        x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        patched_slice = traceable_ops._make_patched_slice()
        with self.assertRaises(NotImplementedError):
            patched_slice(x, [torch.tensor(0), torch.tensor(1), 0], [1, 1, 5])

    # ------------------------------------------------------------------
    # take
    # ------------------------------------------------------------------
    def test_take_embedding_lookup_matches_original(self):
        from keras.src.backend.torch import numpy as torch_backend_numpy

        table = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        indices = torch.tensor([0, 2, 4, 1])
        original = torch_backend_numpy.take(table, indices, axis=0)
        patched = traceable_ops._patched_take(table, indices, axis=0)
        self.assertAllClose(_to_np(original), _to_np(patched))

    def test_take_matches_original_various_axes(self):
        from keras.src.backend.torch import numpy as torch_backend_numpy

        x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        cases = [
            (torch.tensor([0, 2]), 0),
            (torch.tensor([0, 3, 1]), 1),
            (torch.tensor([4, 0]), 2),
            (torch.tensor([-1, -2]), 2),  # negative indices
        ]
        for indices, axis in cases:
            with self.subTest(axis=axis, indices=indices.tolist()):
                original = torch_backend_numpy.take(x, indices, axis=axis)
                patched = traceable_ops._patched_take(x, indices, axis=axis)
                self.assertAllClose(_to_np(original), _to_np(patched))

    def test_take_no_axis_matches_original(self):
        from keras.src.backend.torch import numpy as torch_backend_numpy

        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        indices = torch.tensor([0, 5, 11])
        original = torch_backend_numpy.take(x, indices, axis=None)
        patched = traceable_ops._patched_take(x, indices, axis=None)
        self.assertAllClose(_to_np(original), _to_np(patched))

    def test_take_no_axis_negative_index_matches_flattened_semantics(self):
        """Regression test: ``axis=None`` + negative indices on a
        multi-dim input must wrap against the *flattened* length, not the
        first (unflattened) dimension's size.

        Note this deliberately does **not** compare against
        ``keras.src.backend.torch.numpy.take`` (the pattern every other
        ``test_take_*`` case in this file uses): that upstream Keras
        function has the exact same bug (it computes
        ``x_dim = x.shape[0]`` before flattening for ``axis=None`), so it
        is not a valid reference here. Ground truth is real numpy's
        ``np.take(..., axis=None)``, which flattens first -- e.g. for a
        ``3x4`` input (12 elements), index ``-1`` must resolve to the last
        flattened element (value ``11``), not wrap as if against a
        first-dimension size of 3 (which would incorrectly give ``2``).
        """
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        indices = torch.tensor([-1, -2])

        expected = np.take(_to_np(x), _to_np(indices), axis=None)
        patched = traceable_ops._patched_take(x, indices, axis=None)
        self.assertAllClose(expected, _to_np(patched))

    # ------------------------------------------------------------------
    # scatter_update
    # ------------------------------------------------------------------
    def test_scatter_update_matches_original(self):
        from keras.src.backend.torch import core as torch_core

        inputs = torch.zeros((4, 4), dtype=torch.float32)
        indices = torch.tensor([[0, 0], [1, 1], [2, 2]])
        updates = torch.tensor([10.0, 20.0, 30.0])
        for reduction in (None, "add", "max", "min", "mul"):
            with self.subTest(reduction=reduction):
                original = torch_core.scatter_update(
                    inputs, indices, updates, reduction=reduction
                )
                patched = traceable_ops._patched_scatter_update(
                    inputs, indices, updates, reduction=reduction
                )
                self.assertAllClose(_to_np(original), _to_np(patched))

    def test_scatter_update_overlapping_indices_matches_original(self):
        # Duplicate indices exercise the accumulation/reduction semantics
        # more thoroughly than the diagonal case above.
        from keras.src.backend.torch import core as torch_core

        inputs = torch.full((3, 3), 2.0, dtype=torch.float32)
        indices = torch.tensor([[0, 0], [0, 0], [1, 1]])
        updates = torch.tensor([3.0, 5.0, 7.0])
        for reduction in (None, "add", "max", "min", "mul"):
            with self.subTest(reduction=reduction):
                original = torch_core.scatter_update(
                    inputs, indices, updates, reduction=reduction
                )
                patched = traceable_ops._patched_scatter_update(
                    inputs, indices, updates, reduction=reduction
                )
                self.assertAllClose(_to_np(original), _to_np(patched))

    def test_scatter_update_rejects_unsupported_reduction(self):
        inputs = torch.zeros((2, 2), dtype=torch.float32)
        indices = torch.tensor([[0, 0]])
        updates = torch.tensor([1.0])
        with self.assertRaises(ValueError):
            traceable_ops._patched_scatter_update(
                inputs, indices, updates, reduction="bogus"
            )

    # ------------------------------------------------------------------
    # arange
    # ------------------------------------------------------------------
    def test_arange_matches_original_values(self):
        from keras.src.backend.torch import numpy as torch_backend_numpy

        cases = [
            (0, 10, 1),
            (2, 20, 3),
            (0, 5, None),
            (0.0, 5.0, 0.5),  # float range: dtype should not be forced
        ]
        for start, stop, step in cases:
            with self.subTest(start=start, stop=stop, step=step):
                original = torch_backend_numpy.arange(
                    start, stop=stop, step=step
                )
                patched = traceable_ops._patched_arange(
                    start, stop=stop, step=step
                )
                self.assertAllClose(_to_np(original), _to_np(patched))

    def test_arange_forces_int32_for_integer_ranges(self):
        original = traceable_ops._ORIGINAL_ARANGE(0, stop=10, step=1)
        patched = traceable_ops._patched_arange(0, stop=10, step=1)
        self.assertAllClose(_to_np(original), _to_np(patched))
        self.assertEqual(patched.dtype, torch.int32)

    def test_arange_explicit_dtype_passthrough(self):
        from keras.src.backend.torch import numpy as torch_backend_numpy

        original = torch_backend_numpy.arange(
            0, stop=10, step=1, dtype=torch.int64
        )
        patched = traceable_ops._patched_arange(
            0, stop=10, step=1, dtype=torch.int64
        )
        self.assertAllClose(_to_np(original), _to_np(patched))
        self.assertEqual(patched.dtype, torch.int64)

    # ------------------------------------------------------------------
    # dot_product_attention
    # ------------------------------------------------------------------
    def _reference_attention_inputs(self, num_query_heads, num_kv_heads):
        rng = np.random.default_rng(0)
        batch, q_len, kv_len, head_dim = 2, 3, 4, 8
        query = torch.tensor(
            rng.standard_normal((batch, q_len, num_query_heads, head_dim)),
            dtype=torch.float32,
        )
        key = torch.tensor(
            rng.standard_normal((batch, kv_len, num_kv_heads, head_dim)),
            dtype=torch.float32,
        )
        value = torch.tensor(
            rng.standard_normal((batch, kv_len, num_kv_heads, head_dim)),
            dtype=torch.float32,
        )
        return query, key, value

    def test_dot_product_attention_matches_original_mha(self):
        """Standard multi-head attention (num_query_heads == num_kv_heads)."""
        from keras.src.backend.torch import nn as torch_backend_nn

        query, key, value = self._reference_attention_inputs(4, 4)
        original = torch_backend_nn.dot_product_attention(
            query, key, value, is_causal=True
        )
        patched = traceable_ops._traceable_dot_product_attention(
            query, key, value, is_causal=True
        )
        self.assertAllClose(
            _to_np(original), _to_np(patched), atol=1e-5, rtol=1e-5
        )

    def test_dot_product_attention_gqa_matches_original(self):
        """Grouped-query attention (num_query_heads > num_kv_heads).

        This is the head configuration used by every tiny test model in
        ``export_test.py`` (e.g. ``num_query_heads=4,
        num_key_value_heads=1``). The original Keras torch-backend op
        internally repeats the key/value heads to match the query heads;
        this checks that the traceable replacement -- which does a plain
        batched matmul with no explicit GQA broadcast -- still matches, i.e.
        that its caller must pre-broadcast key/value to the query head count.
        """
        from keras.src.backend.torch import nn as torch_backend_nn

        query, key, value = self._reference_attention_inputs(4, 1)
        original = torch_backend_nn.dot_product_attention(
            query, key, value, is_causal=True
        )

        # `_traceable_dot_product_attention` does not itself broadcast
        # mismatched head counts (unlike the original), so the caller must
        # repeat key/value to the query head count first, exactly as
        # `GemmaAttention._compute_attention`'s non-fused path (used on CPU)
        # does via a manual einsum reshape rather than relying on this op.
        groups = query.shape[2] // key.shape[2]
        key_r = key.repeat_interleave(groups, dim=2)
        value_r = value.repeat_interleave(groups, dim=2)
        patched = traceable_ops._traceable_dot_product_attention(
            query, key_r, value_r, is_causal=True
        )
        self.assertAllClose(
            _to_np(original), _to_np(patched), atol=1e-5, rtol=1e-5
        )

    def test_dot_product_attention_with_mask_matches_original(self):
        from keras.src.backend.torch import nn as torch_backend_nn

        query, key, value = self._reference_attention_inputs(2, 2)
        batch, q_len, _, _ = query.shape
        kv_len = key.shape[1]
        mask = torch.ones((batch, q_len, kv_len), dtype=torch.bool)
        mask[:, :, -1] = False  # mask out the last key position

        original = torch_backend_nn.dot_product_attention(
            query, key, value, mask=mask
        )
        patched = traceable_ops._traceable_dot_product_attention(
            query, key, value, mask=mask
        )
        self.assertAllClose(
            _to_np(original), _to_np(patched), atol=1e-5, rtol=1e-5
        )

    def test_dot_product_attention_soft_cap_matches_manual_reference(self):
        """Attention-logit soft-capping (used by Gemma-family models).

        The upstream Keras torch-backend ``dot_product_attention`` accepts
        an ``attn_logits_soft_cap`` parameter but never applies it (it is
        always routed through ``torch.nn.functional.
        scaled_dot_product_attention``, which has no soft-cap support), so
        there is no meaningful "original" to compare against here. Instead,
        verify the patched implementation directly against a manual
        matmul/softmax/matmul reference with the soft-cap formula applied by
        hand.
        """
        query, key, value = self._reference_attention_inputs(2, 2)
        cap = 5.0

        scale = float(query.shape[-1]) ** -0.5
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = torch.tanh(scores / cap) * cap
        attn = torch.softmax(scores, dim=-1)
        expected = torch.matmul(attn, v).transpose(1, 2)

        patched = traceable_ops._traceable_dot_product_attention(
            query, key, value, attn_logits_soft_cap=cap
        )
        self.assertAllClose(
            _to_np(expected), _to_np(patched), atol=1e-5, rtol=1e-5
        )
