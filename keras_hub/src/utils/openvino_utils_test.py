import unittest.mock

import keras
import numpy as np
import openvino as ov
import pytest
from openvino import Core

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.openvino_utils import compile_model
from keras_hub.src.utils.openvino_utils import get_device
from keras_hub.src.utils.openvino_utils import get_outputs
from keras_hub.src.utils.openvino_utils import ov_infer


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
        device = get_device()
        self.assertIn(device, ["GPU", "CPU"])

        core = Core()
        self.assertIn(device, core.available_devices)

    def test_get_device_consistency(self):
        device1 = get_device()
        device2 = get_device()
        self.assertEqual(device1, device2)

    def test_compile_model_basic_and_precision_hints(self):
        class _MockParam:
            def __init__(self):
                self.output = unittest.mock.MagicMock()
                self.output.get_node.return_value = unittest.mock.MagicMock()

        class _MockOutput:
            def __init__(self):
                self.output = unittest.mock.MagicMock()

        with (
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.ov.Model"
            ) as mock_model_class,
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.get_core"
            ) as mock_get_core,
        ):
            mock_model_class.return_value = unittest.mock.MagicMock()
            mock_core = unittest.mock.MagicMock()
            mock_get_core.return_value = mock_core
            mock_core.compile_model.return_value = unittest.mock.MagicMock()

            struct_params = [_MockParam(), _MockParam()]
            struct_outputs = [_MockOutput()]
            device = "CPU"

            for dtype in ("f32", "f16"):
                with self.subTest(dtype=dtype):
                    result = compile_model(
                        struct_params, struct_outputs, device, dtype
                    )
                    self.assertIsNotNone(result)

            self.assertEqual(mock_core.compile_model.call_count, 2)

    def test_get_outputs_basic_functionality(self):
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
                input_data = flatten_inputs[0]
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
        expected = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(outputs, expected)

    def test_ov_infer_model_caching(self):
        current_device = get_device()

        class MockModel:
            def __init__(self):
                self.dtype = "float32"
                self.ov_compiled_model = unittest.mock.MagicMock()
                self.ov_device = current_device
                self.struct_outputs = ["mock_output"]

            def _parameterize_data(self, inputs):
                return ["mock_param"]

            def _unpack_singleton(self, x):
                return x[0] if len(x) == 1 else x

        def mock_fn(struct_params, stop_token_ids):
            return ["mock_output"]

        model = MockModel()
        test_input = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]
        cached_model = model.ov_compiled_model

        with unittest.mock.patch(
            "keras_hub.src.utils.openvino_utils.get_outputs"
        ) as mock_get_outputs:
            mock_get_outputs.return_value = np.array(
                [[2.0, 4.0, 6.0]], dtype=np.float32
            )
            result = ov_infer(model, test_input, None, mock_fn)

        self.assertIs(model.ov_compiled_model, cached_model)
        self.assertIsNotNone(result)

    def test_ov_infer_dtype_selection(self):
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

        test_cases = [
            ("float32", "f32"),
            ("float16", "f16"),
            ("bfloat16", "f16"),
        ]
        for model_dtype, expected_ov_dtype in test_cases:
            with self.subTest(dtype=model_dtype):
                model = MockModel(model_dtype)
                test_input = [np.array([[1.0, 2.0]], dtype=np.float32)]
                with (
                    unittest.mock.patch(
                        "keras_hub.src.utils.openvino_utils.compile_model"
                    ) as mock_compile,
                    unittest.mock.patch(
                        "keras_hub.src.utils.openvino_utils.get_outputs"
                    ) as mock_get_outputs,
                ):
                    mock_compile.return_value = "mock_compiled_model"
                    mock_get_outputs.return_value = np.array(
                        [[1.0, 2.0]], dtype=np.float32
                    )
                    ov_infer(model, test_input, None, mock_fn)
                    args, kwargs = mock_compile.call_args
                    self.assertEqual(args[3], expected_ov_dtype)
