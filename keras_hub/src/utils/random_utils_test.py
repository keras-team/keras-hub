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
