import hashlib
import os
import subprocess
import sys

import numpy as np

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils import random_utils as random_utils_module
from keras_hub.src.utils.random_utils import record_rng
from keras_hub.src.utils.random_utils import stable_hash


class StableHashTest(TestCase):
    def test_hash_is_deterministic(self):
        self.assertEqual(stable_hash(["a", "b"]), stable_hash(["a", "b"]))

    def test_hash_distinguishes_token_boundaries(self):
        self.assertNotEqual(stable_hash(["ab", "c"]), stable_hash(["a", "bc"]))

    def test_hash_handles_bytes_ints_and_strings(self):
        self.assertEqual(stable_hash([b"a", b"b"]), stable_hash(["a", "b"]))
        self.assertNotEqual(stable_hash([1, 2]), stable_hash([2, 1]))

    def test_hash_is_stable_across_processes(self):
        # `PYTHONHASHSEED` is randomized per process by default, which is
        # exactly what would break Grain worker determinism if we used the
        # builtin `hash()`.
        module_path = random_utils_module.__file__
        script = (
            "import importlib.util;"
            f"spec = importlib.util.spec_from_file_location('ru', "
            f"{module_path!r});"
            "module = importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(module);"
            "print(module.stable_hash(['Hey', 'I', 'like']))"
        )
        env = dict(os.environ, PYTHONHASHSEED="random")
        outputs = set()
        for _ in range(2):
            outputs.add(
                subprocess.check_output(
                    [sys.executable, "-c", script], env=env
                ).strip()
            )
        self.assertLen(outputs, 1)
        self.assertEqual(
            outputs.pop().decode("utf-8"),
            str(stable_hash(["Hey", "I", "like"])),
        )

    def test_hash_matches_blake2b(self):
        expected = hashlib.blake2b(b"a\x00b\x00", digest_size=8).digest()
        self.assertEqual(
            stable_hash(["a", "b"]), int.from_bytes(expected, "big")
        )

    def test_hash_is_independent_of_int_container(self):
        # The same numbers must hash the same whether they arrive as python
        # ints, as a numpy array, or as numpy scalars. Hashing `repr()`
        # made the hash depend on the container, and on the numpy major
        # version, since `repr(np.int64(1))` is `'1'` on numpy 1 and
        # `'np.int64(1)'` on numpy 2.
        expected = stable_hash([1, 2, 3])
        self.assertEqual(stable_hash(np.array([1, 2, 3])), expected)
        self.assertEqual(
            stable_hash([np.int64(1), np.int64(2), np.int64(3)]), expected
        )
        self.assertEqual(
            stable_hash([np.int32(1), np.int32(2), np.int32(3)]), expected
        )

    def test_hash_is_independent_of_float_container(self):
        expected = stable_hash([0.5, 1.5])
        self.assertEqual(stable_hash(np.array([0.5, 1.5])), expected)
        self.assertEqual(
            stable_hash([np.float64(0.5), np.float64(1.5)]), expected
        )
        self.assertEqual(
            stable_hash([np.float32(0.5), np.float32(1.5)]), expected
        )
        # Floats hash by value, so `np.float32(0.1)` and `0.1`, which are
        # different numbers, deliberately keep different hashes.
        self.assertNotEqual(stable_hash([np.float32(0.1)]), stable_hash([0.1]))

    def test_hash_treats_booleans_as_ints(self):
        # `bool` subclasses `int` and `np.bool_.item()` returns a `bool`, so
        # booleans hash as `1` and `0`.
        self.assertEqual(stable_hash([True, False]), stable_hash([1, 0]))
        self.assertEqual(
            stable_hash([np.bool_(True), np.bool_(False)]),
            stable_hash([True, False]),
        )

    def test_string_and_bytes_hashes_are_pinned(self):
        # Numeric tokens hash differently than they used to, but string and
        # bytes records must not move at all: every documented example and
        # every pinned test output depends on these exact digests.
        self.assertEqual(stable_hash([]), 16476032584258269876)
        self.assertEqual(stable_hash(["a", "b"]), 12479399744289587911)
        self.assertEqual(stable_hash([b"a", b"b"]), 12479399744289587911)
        self.assertEqual(stable_hash(["Hey", "I", "like"]), 6955291194717391682)
        self.assertEqual(
            stable_hash(["Keras", "and", "Tensorflow"]), 2804354494843813881
        )
        self.assertEqual(stable_hash(["héllo", "wörld"]), 1464246600793220289)


class RecordRngTest(TestCase):
    def test_same_record_same_stream(self):
        first = record_rng(42, ["a", "b"]).integers(0, 1000, size=8)
        second = record_rng(42, ["a", "b"]).integers(0, 1000, size=8)
        self.assertAllEqual(first, second)

    def test_different_record_different_stream(self):
        first = record_rng(42, ["a", "b"]).integers(0, 1000, size=8)
        second = record_rng(42, ["a", "c"]).integers(0, 1000, size=8)
        self.assertNotAllEqual(first, second)

    def test_different_seed_different_stream(self):
        first = record_rng(42, ["a", "b"]).integers(0, 1000, size=8)
        second = record_rng(43, ["a", "b"]).integers(0, 1000, size=8)
        self.assertNotAllEqual(first, second)

    def test_returns_numpy_generator(self):
        self.assertIsInstance(record_rng(42, ["a"]), np.random.Generator)

    def test_negative_seed(self):
        # `np.random.SeedSequence` rejects negative entropy, so a negative
        # seed used to raise `ValueError: expected non-negative integer`.
        first = record_rng(-7, ["a", "b"]).integers(0, 1000, size=8)
        second = record_rng(-7, ["a", "b"]).integers(0, 1000, size=8)
        self.assertAllEqual(first, second)
        folded = record_rng(-7 & 0xFFFFFFFF, ["a", "b"])
        self.assertAllEqual(first, folded.integers(0, 1000, size=8))
        # Numpy integer seeds fold the same way. Masking one directly would
        # overflow, since `0xFFFFFFFF` does not fit in an `int32`.
        numpy_seed = record_rng(np.int32(-7), ["a", "b"])
        self.assertAllEqual(first, numpy_seed.integers(0, 1000, size=8))

    def test_non_negative_seeds_are_unfolded(self):
        # Folding is the identity below 2**32. The default layer seed is
        # `random.randint(1, int(1e9))`, so this is the common path and its
        # stream must stay exactly what it was before folding was added.
        for seed in (0, 1, 42, 1337, 999999999, 2**32 - 1):
            expected = np.random.default_rng(
                [seed, stable_hash(["a", "b"])]
            ).integers(0, 1000, size=8)
            self.assertAllEqual(
                record_rng(seed, ["a", "b"]).integers(0, 1000, size=8),
                expected,
            )
