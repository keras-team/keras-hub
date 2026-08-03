import unittest
import unittest.mock

import keras
import numpy as np
import torch
from keras.src.backend.torch import core as torch_core
from keras.src.backend.torch import nn as torch_backend_nn
from keras.src.backend.torch import numpy as torch_backend_numpy

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.litertlm import traceable_ops


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# Parity-only: patched ops are called directly on eager tensors (no
# torch.export) and checked against the original Keras implementations.
@unittest.skipUnless(
    keras.config.backend() == "torch",
    "The litertlm traceable-op patches only exist for the PyTorch backend.",
)
class TraceableOpsParityTest(TestCase):
    def test_one_hot_matches_original(self):
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
        with self.assertRaises(ValueError):
            torch_backend_nn.one_hot(torch.tensor([0]), 4, sparse=True)
        with self.assertRaises(ValueError):
            traceable_ops._patched_one_hot(torch.tensor([0]), 4, sparse=True)

    def test_repeat_matches_original(self):
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
        # A 0-D tensor `repeats` is not a plain Python `int`, so Keras's
        # torch-backend `repeat` falls through to the `repeat_interleave`
        # path, while `_traceable_repeat`'s `_is_scalar_integer` check takes
        # the unsqueeze+expand+reshape fast path instead. Values must match.
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
        x = torch.arange(6, dtype=torch.float32)
        original = torch_backend_numpy.repeat(x, 3, axis=None)
        patched = traceable_ops._traceable_repeat(x, 3, axis=None)
        self.assertAllClose(_to_np(original), _to_np(patched))

    def test_repeat_list_repeats_with_axis_delegates_to_original(self):
        # Keras's torch-backend `repeat` rejects a Python list `repeats`
        # with ValueError; inside the patch scope the fallback must
        # delegate to the captured original (mirroring that raise), not
        # recurse into the patch.
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        with self.assertRaises(ValueError):
            torch_backend_numpy.repeat(x, [1, 2], axis=0)
        with traceable_ops._traceable_repeat_scope():
            with self.assertRaises(ValueError):
                torch_backend_numpy.repeat(x, [1, 2], axis=0)

    def test_amax_matches_original(self):
        cases = [
            (torch.randn(2, 3, 4, 5), -1, True),  # 4-D last axis, keepdims
            (torch.randn(2, 3, 4, 5), -1, False),  # 4-D last axis, no keepdims
            (torch.randn(2, 3, 4, 5), 1, True),  # 4-D middle axis
            (torch.randn(2, 3, 4, 5), 0, False),  # 4-D leading axis
            (torch.randn(2, 4, 5), -1, True),  # 3-D (fallthrough)
            (torch.randn(4, 5), -1, True),  # 2-D (fallthrough)
            (torch.randn(7), 0, True),  # 1-D (fallthrough)
            (torch.randn(2, 2, 3, 4, 5), -1, True),  # 5-D (fallthrough)
            (torch.randn(2, 3, 4, 5), (2, 3), True),  # 4-D tuple (fallthrough)
            (torch.randn(2, 3, 4, 5), np.int64(-1), True),  # np.int64 axis
            (torch.randn(2, 3, 4, 5), np.int32(1), True),  # np.int32 axis
        ]
        for x, axis, keepdims in cases:
            with self.subTest(
                shape=tuple(x.shape), axis=axis, keepdims=keepdims
            ):
                patched = traceable_ops._patched_amax(
                    x, axis=axis, keepdims=keepdims
                )
                # The original Keras amax cannot handle NumPy integer axes
                # (raises "truth value of an empty array is ambiguous"); for
                # those cases verify only that the patched version works and
                # matches torch.max directly.
                if isinstance(axis, np.integer):
                    expected = torch.max(
                        x, dim=int(axis), keepdim=keepdims
                    ).values
                    self.assertTrue(torch.equal(expected, patched))
                else:
                    original = torch_backend_numpy.amax(
                        x, axis=axis, keepdims=keepdims
                    )
                    self.assertEqual(
                        tuple(original.shape), tuple(patched.shape)
                    )
                    # The reformulation is the identical reduction; values must
                    # be bit-identical, not merely close (a silent numeric
                    # drift here would corrupt attention-sink softmax
                    # stabilization).
                    self.assertTrue(torch.equal(original, patched))

    def test_amax_public_ops_max_4d_matches(self):
        # The public keras op that GPT-OSS attention actually calls.
        x = torch.randn(2, 3, 4, 5)
        ref = keras.ops.max(x, axis=-1, keepdims=True)
        with unittest.mock.patch.object(
            torch_backend_numpy, "amax", traceable_ops._patched_amax
        ):
            got = keras.ops.max(x, axis=-1, keepdims=True)
        self.assertTrue(torch.equal(ref, got))


# Every module/attribute/replacement triple is resolved lazily, when a
# scope is entered, so nothing above this point would notice a patch aimed
# at the wrong module or a misspelled attribute name.

@unittest.skipUnless(
    keras.config.backend() == "torch",
    "The litertlm traceable-op patches only exist for the PyTorch backend.",
)
class TraceableOpsScopeTest(TestCase):
    PATCHES = [
        (torch_backend_nn, "one_hot", "_patched_one_hot"),
        (torch_backend_numpy, "repeat", "_traceable_repeat"),
        (torch_backend_numpy, "amax", "_patched_amax"),
    ]

    def test_scope_installs_every_replacement_and_restores_it(self):
        originals = [
            (module, attr, getattr(module, attr))
            for module, attr, _ in self.PATCHES
        ]
        with traceable_ops.traceable_ops_scope():
            for module, attr, replacement_name in self.PATCHES:
                with self.subTest(attr=attr, phase="inside"):
                    self.assertIs(
                        getattr(module, attr),
                        getattr(traceable_ops, replacement_name),
                    )
        for module, attr, original in originals:
            with self.subTest(attr=attr, phase="restored"):
                self.assertIs(getattr(module, attr), original)
