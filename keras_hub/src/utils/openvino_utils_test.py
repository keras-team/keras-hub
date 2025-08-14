import unittest.mock

import keras
import numpy as np
import pytest

from keras_hub.src.tests.test_case import TestCase

try:
    import openvino as ov
    from openvino import Core

    from keras_hub.src.utils.openvino_utils import compile_model
    from keras_hub.src.utils.openvino_utils import get_device
    from keras_hub.src.utils.openvino_utils import get_outputs
    from keras_hub.src.utils.openvino_utils import ov_infer
except ImportError:
    ov = None
    Core = None


@pytest.mark.skipif(
    keras.config.backend() != "openvino",
    reason="OpenVINO tests only run with OpenVINO backend",
)
class OpenVINOUtilsTest(TestCase):
    def setUp(self):
        super().setUp()
        if ov is None:
            self.skipTest("OpenVINO not available")

    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a valid device."""
        device = get_device()

        self.assertIn(device, ["GPU", "CPU"])

        core = Core()
        self.assertIn(device, core.available_devices)

    def test_get_device_consistency(self):
        """Test that get_device returns consistent results."""
        device1 = get_device()
        device2 = get_device()

        self.assertEqual(device1, device2)

    def test_compile_model_with_mock_params(self):
        """Test compile_model function interface with mocking."""
        # We mock the OpenVINO components because
        # creating real OpenVINO operations
        # is complex and requires proper parameter
        # graph connections
        with unittest.mock.patch(
            "keras_hub.src.utils.openvino_utils.ov.Model"
        ) as mock_model_class:
            with unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.core"
            ) as mock_core:
                mock_model = unittest.mock.MagicMock()
                mock_model_class.return_value = mock_model
                mock_compiled = unittest.mock.MagicMock()
                mock_core.compile_model.return_value = mock_compiled

                # Mock parameters need both .output
                # and .get_node() for compile_model
                class MockParam:
                    @property
                    def output(self):
                        return self

                    def get_node(self):
                        return unittest.mock.MagicMock()

                class MockOutput:
                    @property
                    def output(self):
                        return unittest.mock.MagicMock()

                param1 = MockParam()
                param2 = MockParam()
                output1 = MockOutput()

                struct_params = [param1, param2]
                struct_outputs = [output1]
                device = "CPU"
                model_dtype = "f32"

                result = compile_model(
                    struct_params, struct_outputs, device, model_dtype
                )

                self.assertIsNotNone(result)
                mock_core.compile_model.assert_called_once()

    def test_compile_model_precision_hints(self):
        """Test compile_model with different precision hints."""
        # Mock the entire compilation process to test precision hint behavior
        with unittest.mock.patch(
            "keras_hub.src.utils.openvino_utils.ov.Model"
        ) as mock_model_class:
            with unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.core"
            ) as mock_core:
                mock_model = unittest.mock.MagicMock()
                mock_model_class.return_value = mock_model
                mock_compiled = unittest.mock.MagicMock()
                mock_core.compile_model.return_value = mock_compiled

                class MockParam:
                    @property
                    def output(self):
                        return self

                    def get_node(self):
                        return unittest.mock.MagicMock()

                class MockOutput:
                    @property
                    def output(self):
                        return unittest.mock.MagicMock()

                param = MockParam()
                output = MockOutput()

                struct_params = [param]
                struct_outputs = [output]
                device = "CPU"

                # Test both f32 and f16 precision to ensure configuration works
                result_f32 = compile_model(
                    struct_params, struct_outputs, device, "f32"
                )
                self.assertIsNotNone(result_f32)

                result_f16 = compile_model(
                    struct_params, struct_outputs, device, "f16"
                )
                self.assertIsNotNone(result_f16)

                self.assertEqual(mock_core.compile_model.call_count, 2)

    def test_get_outputs_basic_functionality(self):
        """Test get_outputs with a mocked compiled model."""

        # Mock result needs to_tuple() method as expected by get_outputs
        class MockResult:
            def __init__(self, data):
                self.data = data

            def to_tuple(self):
                return (self.data,)

        class MockCompiledModel:
            def __init__(self):
                self.inputs = ["input"]
                self.outputs = ["output"]

            def __call__(self, flatten_inputs):
                input_data = flatten_inputs[0]  # flatten_inputs is a list
                # Apply ReLU operation to test inference behavior
                output_data = np.maximum(input_data, 0.0)
                return MockResult(output_data)

        class MockOutput:
            def get_node(self):
                return "mock_relu_node"

        compiled_model = MockCompiledModel()
        struct_outputs = [MockOutput()]

        test_input = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        inputs = [test_input]

        def mock_unpack_singleton(x):
            return x[0] if len(x) == 1 else x

        outputs = get_outputs(
            inputs, struct_outputs, compiled_model, mock_unpack_singleton
        )

        self.assertIsNotNone(outputs)

        # Verify ReLU was applied correctly (negative values become 0)
        expected = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(outputs, expected)

    def test_ov_infer_model_caching(self):
        """Test that ov_infer properly handles model caching attributes."""
        # Use current device to match ov_infer's device check for caching
        current_device = get_device()

        class MockModel:
            def __init__(self):
                self.dtype = "float32"
                self.ov_compiled_model = unittest.mock.MagicMock()
                self.ov_device = (
                    current_device  # Must match for caching to work
                )
                self.struct_outputs = ["mock_output"]

            def _parameterize_data(self, inputs):
                return ["mock_param"]

            def _unpack_singleton(self, x):
                return x[0] if len(x) == 1 else x

        def mock_fn(struct_params, stop_token_ids):
            return ["mock_output"]

        model = MockModel()
        test_input = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]

        # Store reference to verify caching behavior
        cached_model = model.ov_compiled_model

        with unittest.mock.patch(
            "keras_hub.src.utils.openvino_utils.get_outputs"
        ) as mock_get_outputs:
            mock_get_outputs.return_value = np.array(
                [[2.0, 4.0, 6.0]], dtype=np.float32
            )

            result = ov_infer(model, test_input, None, mock_fn)

        # Verify the same compiled model was reused (no recompilation)
        self.assertIs(model.ov_compiled_model, cached_model)
        self.assertIsNotNone(result)

    def test_ov_infer_dtype_selection(self):
        """Test that ov_infer handles different dtypes correctly."""

        class MockModel:
            def __init__(self, dtype):
                self.dtype = dtype
                self.ov_compiled_model = None
                self.ov_device = None
                self.struct_outputs = None

            def _parameterize_data(self, inputs):
                return ["mock_param"]

            def _unpack_singleton(self, x):
                return x[0] if len(x) == 1 else x

        def mock_fn(struct_params, stop_token_ids):
            return ["mock_output"]

        # Test dtype mapping: bfloat16 maps to f16 in OpenVINO
        test_cases = [
            ("float32", "f32"),
            ("float16", "f16"),
            ("bfloat16", "f16"),  # bfloat16 maps to f16 in OpenVINO
        ]

        for model_dtype, expected_ov_dtype in test_cases:
            with self.subTest(dtype=model_dtype):
                model = MockModel(model_dtype)
                test_input = [np.array([[1.0, 2.0]], dtype=np.float32)]

                with unittest.mock.patch(
                    "keras_hub.src.utils.openvino_utils.compile_model"
                ) as mock_compile:
                    with unittest.mock.patch(
                        "keras_hub.src.utils.openvino_utils.get_outputs"
                    ) as mock_get_outputs:
                        mock_compile.return_value = "mock_compiled_model"
                        mock_get_outputs.return_value = np.array(
                            [[1.0, 2.0]], dtype=np.float32
                        )

                        ov_infer(model, test_input, None, mock_fn)

                        mock_compile.assert_called_once()
                        args, kwargs = mock_compile.call_args
                        # Fourth argument is model_dtype passed to compile_model
                        self.assertEqual(args[3], expected_ov_dtype)
