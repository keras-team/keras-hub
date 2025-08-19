import os
import tempfile
import unittest.mock

import keras
import numpy as np
import pytest

from keras_hub.src.tests.test_case import TestCase

try:
    import openvino as ov
    from openvino import Core

    from keras_hub.src.utils.openvino_utils import _contains_training_methods
    from keras_hub.src.utils.openvino_utils import compile_model
    from keras_hub.src.utils.openvino_utils import get_device
    from keras_hub.src.utils.openvino_utils import get_openvino_skip_reason
    from keras_hub.src.utils.openvino_utils import get_outputs
    from keras_hub.src.utils.openvino_utils import load_openvino_supported_tools
    from keras_hub.src.utils.openvino_utils import ov_infer
    from keras_hub.src.utils.openvino_utils import setup_openvino_test_config
    from keras_hub.src.utils.openvino_utils import (
        should_auto_skip_training_test,
    )
except ImportError:
    ov = None
    Core = None


# --- shared test helpers ---
class _MockParam:
    @property
    def output(self):
        return self

    def get_node(self):
        return unittest.mock.MagicMock()


class _MockOutput:
    @property
    def output(self):
        return unittest.mock.MagicMock()


class _MockFspath:
    def __init__(self, path):
        import os

        self.path = path
        self.basename = os.path.basename(path)

    def __str__(self):
        return self.path


class _MockItem:
    def __init__(self, fspath, name):
        self.fspath = (
            _MockFspath(fspath) if not hasattr(fspath, "basename") else fspath
        )
        self.name = name


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
        with (
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.ov.Model"
            ) as mock_model_class,
            unittest.mock.patch(
                "keras_hub.src.utils.openvino_utils.core"
            ) as mock_core,
        ):
            mock_model_class.return_value = unittest.mock.MagicMock()
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

    def test_load_openvino_supported_tools_valid_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as f:
            f.write("keras_hub/src/models/gemma\n")
            f.write("keras_hub/src/models/gpt2\n")
            f.write("keras_hub/src/layers/modeling\n")
            temp_file = f.name

        try:
            result = load_openvino_supported_tools(temp_file)
            expected = [
                "keras_hub/src/models/gemma",
                "keras_hub/src/models/gpt2",
                "keras_hub/src/layers/modeling",
            ]
            self.assertEqual(result, expected)
        finally:
            os.unlink(temp_file)

    def test_load_openvino_supported_tools_nonexistent_file(self):
        result = load_openvino_supported_tools("/nonexistent/file.txt")
        self.assertEqual(result, [])

    def test_load_openvino_supported_tools_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as f:
            temp_file = f.name
        try:
            result = load_openvino_supported_tools(temp_file)
            self.assertEqual(result, [])
        finally:
            os.unlink(temp_file)

    def test_setup_openvino_test_config_openvino_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "openvino_supported_tests.txt")
            with open(config_file, "w") as f:
                f.write("keras_hub/src/models/gemma\n")
                f.write("keras_hub/src/tokenizers\n")

            result = setup_openvino_test_config(temp_dir)
            expected = [
                "keras_hub/src/models/gemma",
                "keras_hub/src/tokenizers",
            ]
            self.assertEqual(result, expected)

    def test_contains_training_methods_with_training_code(self):
        training_code = """
        import keras
        def test_training_method():
            model = keras.Model()
            model.fit(x, y)
            return model
        def test_other_method():
            model.compile(optimizer='adam')
            return model
        """
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        ) as f:
            f.write(training_code)
            temp_file = f.name
        try:
            result = _contains_training_methods(
                temp_file, "test_training_method"
            )
            self.assertTrue(result)
        finally:
            os.unlink(temp_file)

    def test_contains_training_methods_nonexistent_file(self):
        result = _contains_training_methods(
            "/nonexistent/file.py", "test_method"
        )
        self.assertTrue(result)

    def test_should_auto_skip_training_test_non_python_file(self):
        class _SimpleItem:
            def __init__(self, fspath):
                self.fspath = type("MockPath", (), {"basename": fspath})()
                self.name = "test_method"

        item = _SimpleItem("test_file.txt")
        result = should_auto_skip_training_test(item)
        self.assertFalse(result)

    def test_should_auto_skip_training_test_with_training_methods(self):
        training_code = """
        def test_fit_method():
            model.fit(x, y)
            return model
        """
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        ) as f:
            f.write(training_code)
            temp_file = f.name
        try:
            item = _MockItem(temp_file, "test_fit_method")
            result = should_auto_skip_training_test(item)
            self.assertTrue(result)
        finally:
            os.unlink(temp_file)

    def test_get_openvino_skip_reason_specific_test_skip(self):
        class MockItem:
            def __init__(self, test_name):
                self.name = test_name
                self.fspath = type("MockPath", (), {})()
                setattr(self.fspath, "__str__", lambda: "test_file.py")

        expected_reasons = {
            "test_backbone_basics": "Requires trainable backend",
            "test_score_loss": "Non-implemented roll operation",
            "test_layer_behaviors": "Requires trainable backend",
        }
        for test_name, expected_reason in expected_reasons.items():
            item = MockItem(test_name)
            result = get_openvino_skip_reason(item, [], True)
            self.assertEqual(result, expected_reason)

    def test_get_openvino_skip_reason_training_skip(self):
        training_code = """
        def test_training_method():
            model.fit(x, y)
            return model
        """
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        ) as f:
            f.write(training_code)
            temp_file = f.name
        try:
            item = _MockItem(temp_file, "test_training_method")
            result = get_openvino_skip_reason(item, [], True)
            self.assertEqual(result, "Training operations not supported")
        finally:
            os.unlink(temp_file)

    def test_get_openvino_skip_reason_whitelist_supported(self):
        test_path = "/some/path/keras_hub/src/models/gemma/gemma_test.py"
        supported_paths = ["keras_hub/src/models/gemma"]
        item = _MockItem(test_path, "test_inference")
        result = get_openvino_skip_reason(item, supported_paths, False)
        self.assertIsNone(result)

    def test_get_openvino_skip_reason_whitelist_not_supported(self):
        test_path = "/some/path/keras_hub/src/models/gemma3/gemma3_test.py"
        supported_paths = ["keras_hub/src/models/gemma"]
        item = _MockItem(test_path, "test_inference")
        result = get_openvino_skip_reason(item, supported_paths, False)
        self.assertEqual(result, "File/directory not in OpenVINO whitelist")

    def test_get_openvino_skip_reason_no_whitelist(self):
        test_path = "/some/path/keras_hub/src/models/gemma/gemma_test.py"
        item = _MockItem(test_path, "test_inference")
        result = get_openvino_skip_reason(item, [], False)
        self.assertIsNone(result)
