import numpy as np
import openvino as ov
import pytest
from keras import backend
from keras import ops

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.openvino_utils import OPENVINO_DTYPES
from keras_hub.src.utils.openvino_utils import get_outputs
from keras_hub.src.utils.openvino_utils import get_struct_outputs
from keras_hub.src.utils.openvino_utils import is_model_reusable
from keras_hub.src.utils.openvino_utils import parameterize_inputs
from keras_hub.src.utils.openvino_utils import unpack_singleton


@pytest.mark.skipif(
    backend.backend() != "openvino",
    reason="OpenVINO is required for these tests",
)
class TestOpenVinoUtils(TestCase):
    def test_openvino_dtypes_mapping(self):
        self.assertIn("float32", OPENVINO_DTYPES)
        self.assertIn("int32", OPENVINO_DTYPES)
        self.assertIn("bool", OPENVINO_DTYPES)
        self.assertEqual(OPENVINO_DTYPES["float32"], ov.Type.f32)
        self.assertEqual(OPENVINO_DTYPES["int32"], ov.Type.i32)
        self.assertEqual(OPENVINO_DTYPES["bool"], ov.Type.boolean)

    def test_unpack_singleton_single_element_list(self):
        result = unpack_singleton([42])
        self.assertEqual(result, 42)

    def test_unpack_singleton_single_element_tuple(self):
        result = unpack_singleton((42,))
        self.assertEqual(result, 42)

    def test_unpack_singleton_multiple_elements(self):
        input_list = [1, 2, 3]
        result = unpack_singleton(input_list)
        self.assertEqual(result, input_list)

    def test_unpack_singleton_empty_list(self):
        input_list = []
        result = unpack_singleton(input_list)
        self.assertEqual(result, input_list)

    def test_unpack_singleton_non_sequence(self):
        result = unpack_singleton(42)
        self.assertEqual(result, 42)

    def test_parameterize_inputs_numpy_array(self):
        input_array = np.array([1, 2, 3], dtype=np.float32)
        result = parameterize_inputs(input_array)
        self.assertTrue(ops.is_tensor(result))

    def test_parameterize_inputs_different_dtypes(self):
        test_cases = [
            (np.array([1, 2, 3], dtype=np.int32), np.int32),
            (np.array([1.0, 2.0, 3.0], dtype=np.float32), np.float32),
            (np.array([1, 2, 3], dtype=np.int64), np.int64),
        ]

        for input_array, expected_dtype in test_cases:
            result = parameterize_inputs(input_array)
            self.assertTrue(ops.is_tensor(result))

    def test_parameterize_inputs_list(self):
        input_list = [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.int32),
        ]
        result = parameterize_inputs(input_list)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(ops.is_tensor(r) for r in result))

    def test_parameterize_inputs_tuple(self):
        input_tuple = (
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.int32),
        )
        result = parameterize_inputs(input_tuple)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(ops.is_tensor(r) for r in result))

    def test_parameterize_inputs_dict(self):
        input_dict = {
            "a": np.array([1, 2, 3], dtype=np.float32),
            "b": np.array([4, 5, 6], dtype=np.int32),
        }
        result = parameterize_inputs(input_dict)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"a", "b"})
        self.assertTrue(all(ops.is_tensor(v) for v in result.values()))

    def test_parameterize_inputs_integer(self):
        result = parameterize_inputs(42)
        self.assertTrue(ops.is_tensor(result))

    def test_parameterize_inputs_numpy_integer(self):
        result = parameterize_inputs(np.int32(42))
        self.assertTrue(ops.is_tensor(result))

    def test_parameterize_inputs_float(self):
        result = parameterize_inputs(3.14)
        self.assertTrue(ops.is_tensor(result))

    def test_parameterize_inputs_numpy_float(self):
        result = parameterize_inputs(np.float32(3.14))
        self.assertTrue(ops.is_tensor(result))

    def test_parameterize_inputs_unsupported_type(self):
        with self.assertRaisesRegex(TypeError, "Unknown input type"):
            parameterize_inputs("unsupported_string")

    def test_parameterize_inputs_nested_structure(self):
        """Test parameterizing nested structures."""
        nested_input = {
            "list": [np.array([1, 2], dtype=np.float32), 42],
            "dict": {"nested": np.array([3, 4], dtype=np.int32)},
        }
        result = parameterize_inputs(nested_input)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["list"], list)
        self.assertIsInstance(result["dict"], dict)
        self.assertTrue(ops.is_tensor(result["list"][0]))
        self.assertTrue(ops.is_tensor(result["list"][1]))
        self.assertTrue(ops.is_tensor(result["dict"]["nested"]))

    def test_get_struct_outputs(self):
        inputs = np.array([1, 2, 3], dtype=np.float32)
        stop_token_ids = [0, 1]

        def mock_fn(params, stop_tokens):
            return params  # Simple mock that returns the params

        struct_params, struct_outputs = get_struct_outputs(
            inputs, stop_token_ids, mock_fn
        )
        self.assertTrue(ops.is_tensor(struct_params))
        self.assertTrue(ops.is_tensor(struct_outputs))

    def test_get_outputs_with_tensor_input_raises_error(self):
        inputs = [ops.convert_to_tensor(np.array([1, 2, 3]))]
        struct_outputs = np.array([1, 2, 3])

        def mock_compile_model(inputs):
            class MockResult:
                def to_tuple(self):
                    return (np.array([1, 2, 3]),)

            return MockResult()

        with self.assertRaisesRegex(
            ValueError, "inputs should be numpy arrays"
        ):
            get_outputs(inputs, struct_outputs, mock_compile_model)

    def test_get_outputs_with_valid_inputs(self):
        inputs = [np.array([1, 2, 3], dtype=np.float32)]
        struct_outputs = np.array([1, 2, 3])

        def mock_compile_model(inputs):
            class MockResult:
                def to_tuple(self):
                    return (np.array([4, 5, 6]),)

            return MockResult()

        result = get_outputs(inputs, struct_outputs, mock_compile_model)
        self.assertIsInstance(result, np.ndarray)
        self.assertAllClose(result, np.array([4, 5, 6]))

    def test_get_outputs_with_nested_inputs(self):
        inputs = {
            "a": np.array([1, 2], dtype=np.float32),
            "b": [np.array([3, 4], dtype=np.int32)],
        }
        struct_outputs = [np.array([1, 2]), np.array([3, 4])]

        def mock_compile_model(inputs):
            class MockResult:
                def to_tuple(self):
                    return (np.array([5, 6]), np.array([7, 8]))

            return MockResult()

        result = get_outputs(inputs, struct_outputs, mock_compile_model)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertAllClose(result[0], np.array([5, 6]))
        self.assertAllClose(result[1], np.array([7, 8]))

    def test_is_model_reusable_with_matching_token_ids_shape(self):
        inputs = {"token_ids": np.zeros((1, 128))}
        previous_signature = {"token_ids": np.zeros((1, 128))}
        self.assertTrue(is_model_reusable(inputs, previous_signature))

    def test_is_model_reusable_with_mismatched_token_ids_shape(self):
        inputs = {"token_ids": np.zeros((1, 128))}
        previous_signature = {"token_ids": np.zeros((1, 64))}
        self.assertFalse(is_model_reusable(inputs, previous_signature))

    def test_is_model_reusable_with_missing_token_ids(self):
        inputs = {"input_ids": np.zeros((1, 128))}
        previous_signature = {"token_ids": np.zeros((1, 128))}
        with self.assertRaises(NotImplementedError):
            is_model_reusable(inputs, previous_signature)

    def test_is_model_reusable_with_list_input_and_matching_shape(self):
        inputs = [{"token_ids": np.zeros((1, 128))}]
        previous_signature = {"token_ids": np.zeros((1, 128))}
        self.assertTrue(is_model_reusable(inputs, previous_signature))

    def test_is_model_reusable_with_list_input_and_missing_token_ids(self):
        inputs = [{"input_ids": np.zeros((1, 128))}]
        previous_signature = {"token_ids": np.zeros((1, 128))}
        with self.assertRaises(NotImplementedError):
            is_model_reusable(inputs, previous_signature)

    def test_is_model_reusable_with_empty_input_list(self):
        inputs = []
        previous_signature = {"token_ids": np.zeros((1, 128))}
        with self.assertRaises(NotImplementedError):
            is_model_reusable(inputs, previous_signature)

    def test_is_model_reusable_with_non_dict_inputs(self):
        inputs = np.array([1, 2, 3])
        previous_signature = {"token_ids": np.zeros((1, 128))}
        with self.assertRaises(NotImplementedError):
            is_model_reusable(inputs, previous_signature)
