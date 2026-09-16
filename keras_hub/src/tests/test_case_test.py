import grain
import keras
import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types
from keras_hub.src.tests.test_case import grain_ragged_batch
from keras_hub.src.tests.test_case import grain_source_from_tensor_slices


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
