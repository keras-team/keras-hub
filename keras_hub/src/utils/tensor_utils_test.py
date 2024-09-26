import numpy as np
import tensorflow as tf
from keras import ops
from keras import tree

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import any_equal
from keras_hub.src.utils.tensor_utils import convert_preprocessing_inputs
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
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
