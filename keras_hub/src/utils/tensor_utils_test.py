import grain
import numpy as np
import tensorflow as tf
from keras import ops
from keras import tree

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import any_equal
from keras_hub.src.utils.tensor_utils import convert_preprocessing_inputs
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs_grain
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import in_grain_data_pipeline
from keras_hub.src.utils.tensor_utils import is_float_dtype
from keras_hub.src.utils.tensor_utils import is_tensor_type
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import target_gather
from keras_hub.src.utils.tensor_utils import tensor_to_list


class ConvertHelpers(TestCase):
    def test_basics(self):
        inputs = [1, 2, 3]
        # Convert to tf.
        outputs = convert_preprocessing_inputs(inputs)
        self.assertAllEqual(outputs, ops.array(inputs))
        # Convert from tf.
        outputs = convert_preprocessing_outputs(outputs)
        self.assertTrue(is_tensor_type(outputs))
        self.assertAllEqual(outputs, inputs)

    def test_strings(self):
        inputs = ["one", "two"]
        # Convert to tf.
        outputs = convert_preprocessing_inputs(inputs)
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertAllEqual(outputs, tf.constant(inputs))
        # Convert from tf.
        outputs = convert_preprocessing_outputs(outputs)
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs, inputs)

    def test_bytestrings(self):
        inputs = ["one".encode("utf-8"), "two".encode("utf-8")]
        # Convert to tf.
        outputs = convert_preprocessing_inputs(inputs)
        self.assertIsInstance(outputs, tf.Tensor)
        self.assertAllEqual(outputs, tf.constant(inputs))
        # Convert from tf.
        outputs = convert_preprocessing_outputs(outputs)
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs, [x.decode("utf-8") for x in inputs])

    def test_ragged(self):
        inputs = [np.ones((1, 3)), np.ones((1, 2))]
        # Convert to tf.
        outputs = convert_preprocessing_inputs(inputs)
        self.assertIsInstance(outputs, tf.RaggedTensor)
        print(outputs, inputs)
        self.assertAllEqual(outputs, tf.ragged.constant(inputs))
        # Convert from tf.
        outputs = convert_preprocessing_outputs(outputs)
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs, [[[1, 1, 1]], [[1, 1]]])

    def test_composite(self):
        inputs = (
            {
                "text": ["one", "two"],
                "images": [np.ones((4, 4, 3)), np.ones((2, 2, 3))],
                "ragged_ints": [[1, 2], [2, 3, 4]],
            },
            np.array([1, 2]),
            [3, 4],
        )

        outputs = convert_preprocessing_inputs(inputs)
        self.assertIsInstance(outputs[0]["text"], tf.Tensor)
        self.assertIsInstance(outputs[0]["images"], tf.RaggedTensor)
        self.assertIsInstance(outputs[0]["ragged_ints"], tf.RaggedTensor)
        self.assertTrue(is_tensor_type(outputs[1]))
        self.assertTrue(is_tensor_type(outputs[2]))

        outputs = convert_preprocessing_outputs(outputs)
        self.assertIsInstance(outputs[0]["text"], list)
        self.assertIsInstance(outputs[0]["images"], list)
        self.assertIsInstance(outputs[0]["ragged_ints"], list)
        self.assertTrue(is_tensor_type(outputs[1]))
        self.assertTrue(is_tensor_type(outputs[2]))

    def test_python_native_outputs(self):
        inputs = "hello world"
        outputs = convert_preprocessing_outputs(inputs)
        self.assertEqual(outputs, "hello world")
        self.assertIsInstance(outputs, str)
        inputs = ["hello", "world"]
        outputs = convert_preprocessing_outputs(inputs)
        self.assertEqual(outputs, ["hello", "world"])
        self.assertIsInstance(outputs, list)

        def to_list(x):
            return ops.convert_to_numpy(x).tolist() if is_tensor_type(x) else x

        outputs = tree.flatten(tree.map_structure(to_list, outputs))
        inputs = tree.flatten(tree.map_structure(to_list, inputs))
        self.assertAllEqual(outputs, inputs)

    def test_placement(self):
        # Make sure we always place preprocessing on the CPU on all backends.
        @preprocessing_function
        def test(self, inputs):
            for x in inputs:
                if isinstance(x, tf.Tensor):
                    self.assertTrue("CPU" in x.device)
                    self.assertFalse("GPU" in x.device)
            return inputs

        test(self, ([1, 2, 3], ["foo", "bar"], "foo"))


class GrainOutputsTest(TestCase):
    """Preprocessing outputs inside a Grain pipeline must be NumPy/lists.

    Grain pickles elements across worker processes, so backend tensors are
    never returned from a Grain `map`.
    """

    def assertNumpyOutputs(self, x):
        for leaf in tree.flatten(x):
            self.assertNotIsInstance(leaf, tf.Tensor)
            self.assertNotIsInstance(leaf, tf.RaggedTensor)
            if not isinstance(leaf, (str, bytes, int, float, bool)):
                self.assertIsInstance(leaf, np.ndarray)

    def run_grain(self, fn, elements):
        return list(grain.MapDataset.source(elements).map(fn))

    def test_in_grain_data_pipeline_detection(self):
        seen = []
        self.assertFalse(in_grain_data_pipeline())
        self.run_grain(lambda x: seen.append(in_grain_data_pipeline()), [0])
        self.assertEqual(seen, [True])
        self.assertFalse(in_grain_data_pipeline())

    def test_convert_preprocessing_outputs_grain(self):
        inputs = {
            "dense": tf.constant([[1, 2], [3, 4]]),
            "ragged": tf.ragged.constant([[1, 2, 3], [4]]),
            "strings": tf.constant(["one", "two"]),
            "backend": ops.ones((2, 2)),
            "numpy": np.zeros((2,)),
            "scalar_string": "hi",
            "python_list": [1, 2],
            "none": None,
        }
        outputs = convert_preprocessing_outputs_grain(inputs)
        self.assertIsInstance(outputs["dense"], np.ndarray)
        self.assertAllEqual(outputs["dense"], [[1, 2], [3, 4]])
        self.assertEqual(outputs["ragged"], [[1, 2, 3], [4]])
        self.assertEqual(outputs["strings"], ["one", "two"])
        self.assertIsInstance(outputs["backend"], np.ndarray)
        self.assertIsInstance(outputs["numpy"], np.ndarray)
        self.assertEqual(outputs["scalar_string"], "hi")
        self.assertEqual(outputs["python_list"], [1, 2])
        self.assertIsNone(outputs["none"])

    def test_convert_preprocessing_outputs_in_grain(self):
        # Outside grain: backend tensors.
        outputs = convert_preprocessing_outputs(tf.constant([1, 2, 3]))
        self.assertTrue(is_tensor_type(outputs))
        self.assertNotIsInstance(outputs, np.ndarray)
        # Inside grain: numpy.
        (outputs,) = self.run_grain(
            lambda x: convert_preprocessing_outputs(tf.constant(x)),
            [[1, 2, 3]],
        )
        self.assertIsInstance(outputs, np.ndarray)
        self.assertAllEqual(outputs, [1, 2, 3])
        # Ragged and string outputs stay python lists.
        (outputs,) = self.run_grain(
            lambda x: convert_preprocessing_outputs(
                (tf.ragged.constant(x), tf.constant(["a", "b"]))
            ),
            [[[1, 2], [3]]],
        )
        self.assertEqual(outputs, ([[1, 2], [3]], ["a", "b"]))

    def test_convert_preprocessing_outputs_python_in_grain(self):
        # Outside grain: backend tensors.
        outputs = convert_preprocessing_outputs_python(np.array([1, 2, 3]))
        self.assertTrue(is_tensor_type(outputs))
        self.assertNotIsInstance(outputs, np.ndarray)
        # Inside grain: numpy, lists stay lists.
        (outputs,) = self.run_grain(
            lambda x: convert_preprocessing_outputs_python(
                (np.array(x), ops.array(x), [[1], [2, 3]], "hi")
            ),
            [[1, 2, 3]],
        )
        self.assertIsInstance(outputs[0], np.ndarray)
        self.assertIsInstance(outputs[1], np.ndarray)
        self.assertEqual(outputs[2], [[1], [2, 3]])
        self.assertEqual(outputs[3], "hi")

    def test_preprocessing_function_in_grain(self):
        @preprocessing_function
        def fn(self, x):
            return x

        @preprocessing_function
        def fn_with_labels(self, x, y=None, sample_weight=None):
            return x, y, sample_weight

        elements = [
            {"ints": [1, 2, 3], "text": "hello", "image": np.ones((2, 2, 3))}
        ]
        # Direct call: backend tensors.
        outputs = fn(self, elements[0])
        self.assertTrue(is_tensor_type(outputs["ints"]))
        self.assertNotIsInstance(outputs["ints"], np.ndarray)
        # Grain map: numpy and python types.
        (outputs,) = self.run_grain(lambda x: fn(self, x), elements)
        self.assertNumpyOutputs(outputs)
        self.assertAllEqual(outputs["ints"], [1, 2, 3])
        self.assertEqual(outputs["text"], "hello")
        self.assertAllEqual(outputs["image"], np.ones((2, 2, 3)))
        # Grain map with labels.
        (outputs,) = self.run_grain(
            lambda x: fn_with_labels(self, x, [1], [0.5]), elements
        )
        self.assertNumpyOutputs(outputs)
        self.assertAllEqual(outputs[1], [1])
        self.assertAllClose(outputs[2], [0.5])

    def test_grain_batching(self):
        # Grain batches by stacking numpy leaves; outputs must be stackable.
        @preprocessing_function
        def fn(self, x):
            return {"ints": x, "text": tf.strings.upper("hello")}

        ds = grain.MapDataset.source([[1, 2], [3, 4]]).map(
            lambda x: fn(self, x)
        )
        (batch,) = list(ds.batch(2))
        self.assertIsInstance(batch["ints"], np.ndarray)
        self.assertAllEqual(batch["ints"], [[1, 2], [3, 4]])
        self.assertEqual(batch["text"].tolist(), ["HELLO", "HELLO"])

    def test_grain_iter_dataset(self):
        @preprocessing_function
        def fn(self, x):
            return x

        ds = grain.MapDataset.source([[1, 2], [3, 4]]).to_iter_dataset()
        outputs = list(ds.map(lambda x: fn(self, x)))
        self.assertLen(outputs, 2)
        for output in outputs:
            self.assertIsInstance(output, np.ndarray)


class TensorToListTest(TestCase):
    def test_ragged_input(self):
        input_data = tf.ragged.constant([[1, 2], [4, 5, 6]])
        list_output = tensor_to_list(input_data)
        self.assertAllEqual(list_output, [[1, 2], [4, 5, 6]])

    def test_dense_input(self):
        input_data = tf.constant([[1, 2], [3, 4]])
        list_output = tensor_to_list(input_data)
        self.assertAllEqual(list_output, [[1, 2], [3, 4]])

    def test_scalar_input(self):
        input_data = tf.constant(1)
        list_output = tensor_to_list(input_data)
        self.assertEqual(list_output, 1)

    def test_ragged_strings(self):
        input_data = tf.ragged.constant([["▀▁▂▃", "samurai"]])
        detokenize_output = tensor_to_list(input_data)
        self.assertAllEqual(detokenize_output, [["▀▁▂▃", "samurai"]])

    def test_dense_strings(self):
        input_data = tf.constant([["▀▁▂▃", "samurai"]])
        detokenize_output = tensor_to_list(input_data)
        self.assertAllEqual(detokenize_output, [["▀▁▂▃", "samurai"]])

    def test_scalar_string(self):
        input_data = tf.constant("▀▁▂▃")
        detokenize_output = tensor_to_list(input_data)
        self.assertEqual(detokenize_output, "▀▁▂▃")

    def test_string_with_utf8_error(self):
        input_data = tf.constant([b"hello\xf2\xf1\x91\xe5"])
        detokenize_output = tensor_to_list(input_data)
        self.assertEqual(detokenize_output, ["hello"])


class ConvertToRaggedBatch(TestCase):
    def test_convert_1d_tensor(self):
        inputs = tf.constant([1, 2, 3])
        outputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        self.assertIsInstance(outputs, tf.RaggedTensor)
        self.assertAllEqual(outputs, [[1, 2, 3]])
        self.assertTrue(unbatched)
        self.assertTrue(rectangular)

    def test_convert_2d_tensor(self):
        inputs = tf.constant([[1, 2, 3], [1, 2, 3]])
        outputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        self.assertIsInstance(outputs, tf.RaggedTensor)
        self.assertAllEqual(outputs, [[1, 2, 3], [1, 2, 3]])
        self.assertFalse(unbatched)
        self.assertTrue(rectangular)

    def test_convert_ragged(self):
        inputs = tf.ragged.constant([[1, 2], [1]])
        outputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        self.assertIsInstance(outputs, tf.RaggedTensor)
        self.assertAllEqual(outputs, [[1, 2], [1]])
        self.assertFalse(unbatched)
        self.assertFalse(rectangular)


class MaskedAnyEqualTest(tf.test.TestCase):
    def test_basic_equality(self):
        inputs = ops.array([1, 2, 3, 5])
        values = [3, 5]
        padding_mask = ops.array([True, True, True, False])
        expected_output = np.array([False, False, True, False])
        result = any_equal(inputs, values, padding_mask)
        result = ops.convert_to_numpy(result)
        self.assertAllEqual(result, expected_output)

    def test_multiple_values(self):
        inputs = ops.array([2, 4, 7, 9])
        values = [5, 4, 9]
        padding_mask = ops.array([True, True, True, True])
        expected_output = np.array([False, True, False, True])
        result = any_equal(inputs, values, padding_mask)
        result = ops.convert_to_numpy(result)
        self.assertAllEqual(result, expected_output)

    def test_padding_mask(self):
        inputs = ops.array([1, 5, 3, 2])
        values = [5, 3]
        padding_mask = ops.array([True, False, True, False])
        expected_output = np.array([False, False, True, False])
        result = any_equal(inputs, values, padding_mask)
        result = ops.convert_to_numpy(result)
        self.assertAllEqual(result, expected_output)

    def test_input_shaped_values(self):
        inputs = ops.array([1, 5, 3, 2])
        values = [[5, 5, 5, 5], [3, 3, 3, 3]]
        padding_mask = ops.array([True, False, True, False])
        expected_output = np.array([False, False, True, False])
        result = any_equal(inputs, values, padding_mask)
        result = ops.convert_to_numpy(result)
        self.assertAllEqual(result, expected_output)


class TargetGatherTest(TestCase):
    def test_target_gather_boxes_batched(self):
        target_boxes = np.array(
            [[0, 0, 5, 5], [0, 5, 5, 10], [5, 0, 10, 5], [5, 5, 10, 10]]
        )
        target_boxes = ops.expand_dims(target_boxes, axis=0)
        indices = np.array([[0, 2]], dtype="int32")
        expected_boxes = np.array([[0, 0, 5, 5], [5, 0, 10, 5]])
        expected_boxes = ops.expand_dims(expected_boxes, axis=0)
        res = target_gather(target_boxes, indices)
        self.assertAllClose(expected_boxes, res)

    def test_target_gather_boxes_unbatched(self):
        target_boxes = np.array(
            [[0, 0, 5, 5], [0, 5, 5, 10], [5, 0, 10, 5], [5, 5, 10, 10]],
            "int32",
        )
        indices = np.array([0, 2], dtype="int32")
        expected_boxes = np.array([[0, 0, 5, 5], [5, 0, 10, 5]])
        res = target_gather(target_boxes, indices)
        self.assertAllClose(expected_boxes, res)

    def test_target_gather_classes_batched(self):
        target_classes = np.array([[1, 2, 3, 4]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2]], dtype="int32")
        expected_classes = np.array([[1, 3]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_unbatched(self):
        target_classes = np.array([1, 2, 3, 4])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([0, 2], dtype="int32")
        expected_classes = np.array([1, 3])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_batched_with_mask(self):
        target_classes = np.array([[1, 2, 3, 4]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2]], dtype="int32")
        masks = np.array(([[False, True]]))
        masks = ops.expand_dims(masks, axis=-1)
        # the second element is masked
        expected_classes = np.array([[1, 0]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = target_gather(target_classes, indices, masks)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_batched_with_mask_val(self):
        target_classes = np.array([[1, 2, 3, 4]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2]], dtype="int32")
        masks = np.array(([[False, True]]))
        masks = ops.expand_dims(masks, axis=-1)
        # the second element is masked
        expected_classes = np.array([[1, -1]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = target_gather(target_classes, indices, masks, -1)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_unbatched_with_mask(self):
        target_classes = np.array([1, 2, 3, 4])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([0, 2], dtype="int32")
        masks = np.array([False, True])
        masks = ops.expand_dims(masks, axis=-1)
        expected_classes = np.array([1, 0])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = target_gather(target_classes, indices, masks)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_with_empty_targets(self):
        target_classes = np.array([])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([0, 2], dtype="int32")
        # return all 0s since input is empty
        expected_classes = np.array([0, 0])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_multi_batch(self):
        target_classes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2], [1, 3]], dtype="int32")
        expected_classes = np.array([[1, 3], [6, 8]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_invalid_rank(self):
        targets = np.random.normal(size=[32, 2, 2, 2])
        indices = np.array([0, 1], dtype="int32")
        with self.assertRaisesRegex(ValueError, "larger than 3"):
            _ = target_gather(targets, indices)


class IsFloatDtypeTest(TestCase):
    def test_float_dtypes_return_true(self):
        float_dtypes = [
            "float16",
            "float32",
            "float64",
            "bfloat16",
        ]
        for dtype in float_dtypes:
            self.assertTrue(is_float_dtype(dtype))

    def test_non_float_dtypes_return_false(self):
        non_float_dtypes = [
            "int8",
            "int32",
            "uint8",
            "bool",
            "string",
        ]
        for dtype in non_float_dtypes:
            self.assertFalse(is_float_dtype(dtype))
