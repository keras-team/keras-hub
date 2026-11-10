import types
import unittest.mock

import keras
import numpy as np
import openvino as ov
import openvino.opset16 as ov_opset
import pytest
from openvino import Core

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.openvino_utils import compile_model
from keras_hub.src.utils.openvino_utils import get_device
from keras_hub.src.utils.openvino_utils import get_input_signature
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

    def test_ov_infer_dtype_selection(self):
        class MockModel:
            def __init__(self, dtype):
                self.dtype = dtype
                self.ov_compiled_model = None
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
                    ov_infer(model, test_input, mock_fn, static_args=(None,))
                    args, kwargs = mock_compile.call_args
                    self.assertEqual(args[3], expected_ov_dtype)


class OpenVINOCacheTest(TestCase):
    def test_get_input_signature(self):
        x = np.zeros((2, 5), dtype="int32")
        signature = get_input_signature(
            {"token_ids": x, "padding_mask": x.astype("bool")}
        )
        self.assertEqual(signature, (((None, 5), "bool"), ((None, 5), "int32")))

    def test_get_input_signature_ignores_batch_size(self):
        self.assertEqual(
            get_input_signature([np.zeros((1, 5))]),
            get_input_signature([np.zeros((3, 5))]),
        )

    def test_get_input_signature_keeps_rank(self):
        self.assertNotEqual(
            get_input_signature([np.zeros(())]),
            get_input_signature([np.zeros((3,))]),
        )

    def test_get_outputs_keeps_none_outputs(self):
        compiled = unittest.mock.Mock()
        compiled.return_value.to_tuple.return_value = ("a", "b")
        outputs = get_outputs(
            [np.zeros((1,))],
            {"token_ids": 0, "images": None, "padding_mask": 0},
            compiled,
            lambda x: x,
        )
        self.assertEqual(
            outputs, {"token_ids": "b", "images": None, "padding_mask": "a"}
        )

    def test_compile_model_drops_none_outputs(self):
        param = ov_opset.parameter([2], ov.Type.f32)

        def wrap(node):
            return types.SimpleNamespace(output=node.output(0))

        outputs = [
            wrap(ov_opset.relu(param)),
            None,
            wrap(ov_opset.negative(param)),
        ]
        compiled = compile_model([wrap(param)], outputs, "CPU", "f32")
        self.assertEqual(len(compiled.outputs), 2)

    def count_compiles(self, calls):
        class MockModel:
            dtype = "float32"

            def _parameterize_data(self, inputs):
                return ["mock_param"]

            def _unpack_singleton(self, x):
                return x[0] if len(x) == 1 else x

        def mock_fn(struct_params, *static_args):
            return ["mock_output"]

        model = MockModel()
        with (
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.get_device",
                return_value="CPU",
            ),
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.compile_model"
            ) as mock_compile,
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.get_outputs"
            ),
        ):
            for inputs, static_args in calls:
                ov_infer(model, inputs, mock_fn, static_args=static_args)
        return mock_compile.call_count

    def test_ov_infer_reuses_model(self):
        x = [np.zeros((2, 5), dtype="int32")]
        self.assertEqual(self.count_compiles([(x, (None,)), (x, (None,))]), 1)

    def test_ov_infer_reuses_model_across_batch_sizes(self):
        calls = [
            ([np.zeros((1, 5), dtype="int32")], (None,)),
            ([np.zeros((4, 5), dtype="int32")], (None,)),
        ]
        self.assertEqual(self.count_compiles(calls), 1)

    def test_ov_infer_recompiles_on_sequence_length_change(self):
        calls = [
            ([np.zeros((2, 5), dtype="int32")], (None,)),
            ([np.zeros((2, 8), dtype="int32")], (None,)),
        ]
        self.assertEqual(self.count_compiles(calls), 2)

    def test_ov_infer_recompiles_on_static_args_change(self):
        x = [np.zeros((2, 5), dtype="int32")]
        calls = [(x, (None,)), (x, ((2,),))]
        self.assertEqual(self.count_compiles(calls), 2)

    def test_ov_infer_keeps_cache_when_compile_fails(self):
        class MockModel:
            dtype = "float32"

            def _parameterize_data(self, inputs):
                return ["mock_param"]

            def _unpack_singleton(self, x):
                return x

        model = MockModel()
        with (
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.get_device",
                return_value="CPU",
            ),
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.compile_model",
                side_effect=["compiled", RuntimeError("compile failed")],
            ),
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.get_outputs"
            ),
        ):
            ov_infer(model, [np.zeros((2, 5))], lambda p: ["old"])
            signature = model.ov_signature
            with self.assertRaises(RuntimeError):
                ov_infer(model, [np.zeros((2, 8))], lambda p: ["new"])
        self.assertEqual(model.struct_outputs, ["old"])
        self.assertEqual(model.ov_compiled_model, "compiled")
        self.assertEqual(model.ov_signature, signature)
