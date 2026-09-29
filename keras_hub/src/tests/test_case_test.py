import collections
import os
import subprocess
import sys
import unittest
from types import MappingProxyType
from unittest import mock

import grain
import keras
import numpy as np
from keras import ops

from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types
from keras_hub.src.tests.test_case import grain_ragged_batch
from keras_hub.src.tests.test_case import grain_source_from_tensor_slices
from keras_hub.src.utils.tensor_utils import tf


class AssertionsTest(TestCase):
    """The assertions must handle jagged and dict data, not just arrays.

    `np.array()` raises on both, so every case below is one that the plain
    `keras.src.testing.TestCase` implementation cannot express.
    """

    def test_ragged_lists(self):
        self.assertAllEqual(
            [[9, 10, 11, 12], [9, 12]], [[9, 10, 11, 12], [9, 12]]
        )
        self.assertAllClose([[1.0, 2.0], [3.0]], [[1.0, 2.0], [3.0]])

    def test_ragged_lists_detect_mismatch(self):
        with self.assertRaises(AssertionError):
            self.assertAllEqual([[9, 10], [9, 12]], [[9, 10], [9, 13]])
        with self.assertRaises(AssertionError):
            self.assertAllClose([[1.0, 2.0], [3.0]], [[1.0, 2.0], [4.0]])

    def test_ragged_lists_detect_length_mismatch(self):
        with self.assertRaises(AssertionError):
            self.assertAllEqual([[1, 2], [3]], [[1, 2], [3], [4]])
        with self.assertRaises(AssertionError):
            self.assertAllEqual([[1, 2], [3]], [[1, 2], [3, 4]])

    def test_detect_shape_mismatch(self):
        # numpy broadcasts a scalar against an array and calls it equal, so
        # these pass unless shapes are compared first.
        with self.assertRaises(AssertionError):
            self.assertAllEqual(5, [5, 5])
        with self.assertRaises(AssertionError):
            self.assertAllClose(5.0, [5.0, 5.0])
        with self.assertRaises(AssertionError):
            self.assertAllEqual(np.zeros(()), np.zeros((2, 3)))

    def test_detect_nesting_depth_mismatch(self):
        # `[3]` vs `3` is a rank mismatch reached partway down the recursion.
        with self.assertRaises(AssertionError):
            self.assertAllEqual([[1, 2], [3]], [[1, 2], 3])

    @unittest.skipIf(tf is None, "Requires TF")
    def test_ragged_tensors(self):
        ragged = tf.ragged.constant([[9, 10, 11, 12], [9, 12]])
        self.assertAllEqual(ragged, [[9, 10, 11, 12], [9, 12]])
        self.assertAllClose(ragged, [[9, 10, 11, 12], [9, 12]])
        with self.assertRaises(AssertionError):
            self.assertAllEqual(ragged, [[9, 10, 11, 12], [9, 13]])

    def test_dicts(self):
        x = {"token_ids": np.array([[1, 2]]), "padding_mask": [[True, False]]}
        self.assertAllClose(x, dict(x))
        self.assertAllEqual(x, dict(x))

    def test_dicts_detect_mismatch(self):
        x = {"token_ids": np.array([[1, 2]])}
        with self.assertRaises(AssertionError):
            self.assertAllClose(x, {"token_ids": np.array([[1, 3]])})
        with self.assertRaises(AssertionError):
            self.assertAllClose(x, {"other_key": np.array([[1, 2]])})
        with self.assertRaises(AssertionError):
            self.assertAllClose(x, np.array([[1, 2]]))

    def test_nested_dicts_and_tuples(self):
        x = ({"ids": [[1, 2, 3], [4]]}, np.array([0.5, 1.0]))
        self.assertAllClose(x, ({"ids": [[1, 2, 3], [4]]}, [0.5, 1.0]))
        with self.assertRaises(AssertionError):
            self.assertAllClose(x, ({"ids": [[1, 2, 3], [5]]}, [0.5, 1.0]))

    def test_strings(self):
        self.assertAllEqual(["a", "bb"], ["a", "bb"])
        self.assertAllClose([["a"], ["b", "c"]], [["a"], ["b", "c"]])
        if tf is not None:
            self.assertAllEqual(tf.constant(["a", "bb"]), ["a", "bb"])
        self.assertAllEqual(np.array([b"a", b"bb"]), ["a", "bb"])
        with self.assertRaises(AssertionError):
            self.assertAllEqual(["a", "bb"], ["a", "cc"])

    def test_bytes_are_not_equal_to_str(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(b"abc", "abc")
        with self.assertRaises(AssertionError):
            self.assertEqual([b"abc", b"d"], ["abc", "d"])

    def test_bytes_that_are_not_utf8(self):
        # Must compare equal rather than raising `UnicodeDecodeError`.
        self.assertAllEqual([b"\xe4\xbd"], [b"\xe4\xbd"])
        self.assertAllEqual([b"abc"], ["abc"])

    def test_dicts_of_arrays(self):
        self.assertEqual({"ids": np.array([1, 2])}, {"ids": np.array([1, 2])})
        with self.assertRaises(AssertionError):
            self.assertEqual(
                {"x": np.array([1.0, 2.0])}, {"x": np.array([1.0, 2.0000001])}
            )

    def test_dict_keys_of_mixed_types(self):
        self.assertAllEqual({1: 1, "a": 2}, {1: 1, "a": 2})

    def test_zero_dim_and_non_array_leaves(self):
        with self.assertRaises(AssertionError):
            self.assertAllEqual([[1, 2], [3]], np.array(5))
        policy = keras.DTypePolicy("float32")
        with self.assertRaises(AssertionError):
            self.assertNotAllEqual(policy, policy)
        with self.assertRaises(AssertionError):
            self.assertNotAllEqual(2**70, 2**70)

    def test_non_dict_mappings(self):
        # A `MappingProxyType` must be walked like a dict. Comparing the two
        # mappings directly with `==` raises on the array values, so a
        # fallback that does not recurse reports these as unequal.
        self.assertAllClose(
            MappingProxyType({"k": np.array([1.0, 2.0])}),
            MappingProxyType({"k": np.array([1.0, 2.0])}),
        )
        with self.assertRaises(AssertionError):
            self.assertAllClose(
                MappingProxyType({"k": 1.0}), MappingProxyType({"k": 2.0})
            )

    def test_namedtuples_compare_by_field_name(self):
        nt = collections.namedtuple("NT", ["a", "b"])
        nt2 = collections.namedtuple("NT2", ["b", "a"])
        with self.assertRaises(AssertionError):
            self.assertAllClose(
                nt(a=[1.0], b=[2.0, 3.0]), nt2(b=[1.0], a=[2.0, 3.0])
            )

    @unittest.skipIf(tf is None, "Requires TensorFlow")
    def test_ragged_is_not_equal_to_dense(self):
        with self.assertRaises(AssertionError):
            self.assertAllEqual(
                tf.ragged.constant([[1, 2], [3, 4]]),
                tf.constant([[1, 2], [3, 4]]),
            )

    def test_assert_not_all_equal(self):
        self.assertNotAllEqual([1, 2], [1, 3])
        self.assertNotAllEqual([[1, 2], [3]], [[1, 2], [4]])
        with self.assertRaises(AssertionError):
            self.assertNotAllEqual([1, 2], [1, 2])

    def test_assert_all_in_range(self):
        self.assertAllInRange(np.array([0.0, 0.5, 1.0]), 0.0, 1.0)
        self.assertAllInRange(ops.zeros((2, 2)), -1.0, 1.0)
        with self.assertRaises(AssertionError):
            self.assertAllInRange(np.array([0.0, 1.5]), 0.0, 1.0)


class GrainSourceFromTensorSlicesTest(TestCase):
    def test_list_of_strings(self):
        ds = grain_source_from_tensor_slices(["a", "bb"])
        self.assertEqual(list(ds), ["a", "bb"])

    def test_dict_and_tuple_structure(self):
        input_data = (
            {"text": ["a", "bb"], "ids": np.array([[1, 2], [3, 4]])},
            [0, 1],
            ops.convert_to_tensor([0.5, 1.0]),
        )
        ds = grain_source_from_tensor_slices(input_data)
        self.assertEqual(len(ds), 2)
        x, y, sw = ds[1]
        self.assertEqual(x["text"], "bb")
        self.assertAllEqual(x["ids"], [3, 4])
        self.assertIsInstance(x["ids"], np.ndarray)
        self.assertEqual(y, 1)
        self.assertEqual(float(sw), 1.0)
        assert_grain_safe_types(ds[1])

    @unittest.skipIf(tf is None, "Requires TF")
    def test_ragged_and_string_tensors(self):
        ragged = tf.ragged.constant([[1, 2, 3], [4]])
        strings = tf.constant(["a", "bb"])
        ds = grain_source_from_tensor_slices({"x": ragged, "s": strings})
        self.assertEqual(
            list(ds), [{"x": [1, 2, 3], "s": "a"}, {"x": [4], "s": "bb"}]
        )

    def test_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "same first dimension"):
            grain_source_from_tensor_slices((["a", "b"], [1]))


class GrainRaggedBatchTest(TestCase):
    def test_stacks_dense_leaves(self):
        elements = [
            {"a": np.array([1, 2]), "b": 1.0},
            {"a": np.array([3, 4]), "b": 2.0},
        ]
        batch = grain_ragged_batch(elements)
        self.assertAllEqual(batch["a"], [[1, 2], [3, 4]])
        self.assertAllClose(batch["b"], [1.0, 2.0])

    def test_lists_ragged_leaves(self):
        elements = [
            ([1, 2, 3], np.array([1, 2, 3]), "a"),
            ([4], np.array([4]), "bb"),
        ]
        batch = grain_ragged_batch(elements)
        self.assertEqual(
            batch, ([[1, 2, 3], [4]], [[1, 2, 3], [4]], ["a", "bb"])
        )


class AssertGrainSafeTypesTest(TestCase):
    def test_accepts_numpy_and_python(self):
        assert_grain_safe_types(
            {"a": np.ones(2), "b": [[1, 2], [3]], "c": "str", "d": None, "e": 1}
        )

    def test_rejects_backend_and_tf_tensors(self):
        with self.assertRaisesRegex(AssertionError, "NumPy arrays"):
            assert_grain_safe_types({"a": ops.ones(2)})
        if tf is not None:
            with self.assertRaisesRegex(AssertionError, "NumPy arrays"):
                assert_grain_safe_types((tf.ones(2),))


class LeakyLayer(keras.layers.Layer):
    """A layer that returns backend tensors regardless of the pipeline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs):
        return ops.convert_to_tensor(np.asarray(inputs, dtype="int32"))


class GrainHarnessTest(TestCase):
    @unittest.skipIf(tf is None, "Requires TF")
    def test_grain_checks_pass_for_grain_safe_layer(self):
        layer = StartEndPacker(
            sequence_length=4, start_value=1, end_value=2, pad_value=0
        )
        input_data = tf.ragged.constant([[5, 6], [7]])
        expected = [[1, 5, 6, 2], [1, 7, 2, 0]]
        self.run_grain_preprocessing_test(layer, input_data, expected)

    def test_grain_checks_catch_backend_tensor_outputs(self):
        layer = LeakyLayer()
        input_data = np.array([[1, 2], [3, 4]])
        # Sanity check the layer does leak backend tensors inside Grain.
        (element,) = list(grain.MapDataset.source([input_data]).map(layer))
        self.assertTrue(ops.is_tensor(element))
        with self.assertRaisesRegex(AssertionError, "NumPy arrays"):
            self.run_grain_preprocessing_test(layer, input_data, input_data)

    def test_grain_checks_catch_output_mismatch(self):
        layer = StartEndPacker(sequence_length=3, start_value=1)
        input_data = [[5, 6], [7, 8]]
        with self.assertRaises(AssertionError):
            self.run_grain_preprocessing_test(
                layer, input_data, [[1, 5, 6], [1, 7, 7]]
            )


class TestCaseTest(TestCase):
    @unittest.skipIf(
        keras.config.backend() == "tensorflow",
        reason=(
            "The TensorFlow backend cannot boot with an attribute-less "
            "`tensorflow` planted in `sys.modules`."
        ),
    )
    def test_setup_with_empty_tensorflow(self):
        """Pin the `setUp` guard against a partly uninstalled TensorFlow.

        A leftover empty `tensorflow/` directory imports as a PEP 420
        namespace package: the import succeeds, the module has no
        attributes, and Keras' `LazyModule.available` still reports True, so
        `set_random_seed` raises `AttributeError` on `tf.random`.

        A real directory cannot reproduce that while TensorFlow is installed
        -- a namespace portion does not terminate the import search, so the
        regular package wins -- hence the planted module below, in a
        subprocess so it cannot leak into the rest of the session.
        """
        script = """
import importlib.machinery
import random
import sys
import types
import unittest

import numpy as np

# `origin=None` mirrors a PEP 420 namespace package. A bare ModuleType with
# `__spec__ = None` is not equivalent: on torch it makes `find_spec` raise
# `ValueError: tensorflow.__spec__ is None`, a different failure.
_tf_proxy = types.ModuleType("tensorflow")
_tf_proxy.__spec__ = importlib.machinery.ModuleSpec(
    "tensorflow", loader=None, origin=None
)
sys.modules["tensorflow"] = _tf_proxy

import keras

# Liveness check: prove the environment really is broken before testing the
# guard against it. If this stops raising, the guard is no longer exercised
# and the test must fail rather than pass for free.
try:
    keras.utils.set_random_seed(87654321)
except AttributeError:
    pass
else:
    raise AssertionError(
        "Precondition not met: set_random_seed did not raise, so the "
        "attribute-less TensorFlow was not picked up by Keras."
    )

from keras_hub.src.tests.test_case import TestCase


def _draw():
    values = [random.random(), float(np.random.rand())]
    if keras.config.backend() == "torch":
        import torch

        values.append(float(torch.rand(1)))
    return values


class DummyTest(TestCase):
    def test_setup_reseeds_under_broken_tensorflow(self):
        # Two `setUp` calls must draw the same values. That is what fails if
        # the fallback swallows the error without reseeding torch.
        self.setUp()
        first = _draw()
        self.setUp()
        second = _draw()
        # Bare `assert` is stripped under `PYTHONOPTIMIZE`, which the parent
        # propagates via `os.environ.copy()`. Use real assertions.
        self.assertEqual(first, second, "setUp did not reseed")
        if keras.config.backend() == "torch":
            import torch

            # `torch.manual_seed` is the only statement the fallback
            # actually replays, so pin it directly.
            self.assertEqual(torch.initial_seed(), 87654321)


if __name__ == '__main__':
    unittest.main()
"""
        if not sys.executable:
            self.skipTest("sys.executable is not available.")
        tmpdir = self.get_temp_dir()
        script_path = os.path.join(tmpdir, "test_dummy.py")
        with open(script_path, "w") as f:
            f.write(script)

        env = os.environ.copy()
        wt_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )
        # Joining with an empty existing value would leave a trailing
        # separator, which puts the CWD on the child's `sys.path`.
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            wt_path + os.pathsep + existing if existing else wt_path
        )

        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # Pin that the child ran the one test, rather than trusting the
        # exit code alone to tell us the body executed.
        self.assertIn("Ran 1 test", result.stderr)

    def test_setup_propagates_unrelated_attribute_error(self):
        """The guard must not swallow an unrelated `AttributeError`.

        It exists for a partly uninstalled TensorFlow, which is exactly the
        state where `tensor_utils.tf` is None. With TensorFlow healthy, an
        `AttributeError` from inside `set_random_seed` is a real bug, and
        swallowing it would leave the whole suite running unseeded with no
        diagnostic.
        """
        if tf is None:
            self.skipTest("Requires TensorFlow to be installed.")

        def raise_unrelated(seed):
            raise AttributeError("nothing to do with TensorFlow")

        with mock.patch.object(keras.utils, "set_random_seed", raise_unrelated):
            with self.assertRaises(AttributeError):
                self.setUp()
