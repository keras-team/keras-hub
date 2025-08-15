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

    def test_load_openvino_supported_tools_valid_file(self):
        """Test loading supported tools from a valid file."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as f:
            f.write("keras_hub/src/models/gemma\n")
            f.write("keras_hub/src/models/gpt2\n")
            f.write("keras_hub/src/layers/modeling\n")
            f.write("\n")
            f.write("# Comment line\n")
            temp_file = f.name

        try:
            result = load_openvino_supported_tools(temp_file)
            expected = [
                "keras_hub/src/models/gemma",
                "keras_hub/src/models/gpt2",
                "keras_hub/src/layers/modeling",
                "# Comment line",
            ]
            self.assertEqual(result, expected)
        finally:
            os.unlink(temp_file)

    def test_load_openvino_supported_tools_nonexistent_file(self):
        """Test loading supported tools from a nonexistent file."""
        result = load_openvino_supported_tools("/nonexistent/file.txt")
        self.assertEqual(result, [])

    def test_load_openvino_supported_tools_empty_file(self):
        """Test loading supported tools from an empty file."""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as f:
            temp_file = f.name

        try:
            result = load_openvino_supported_tools(temp_file)
            self.assertEqual(result, [])
        finally:
            os.unlink(temp_file)

    def test_setup_openvino_test_config_non_openvino_backend(self):
        """Test setup_openvino_test_config with non-OpenVINO backend."""
        try:
            with unittest.mock.patch(
                "keras.config.backend", return_value="tensorflow"
            ):
                result = setup_openvino_test_config("/some/path")
                self.assertEqual(result, [])
        finally:
            pass

    def test_setup_openvino_test_config_openvino_backend(self):
        """Test setup_openvino_test_config with OpenVINO backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "openvino_supported_tests.txt")
            with open(config_file, "w") as f:
                f.write("keras_hub/src/models/gemma\n")
                f.write("keras_hub/src/tokenizers\n")

            with unittest.mock.patch(
                "keras.config.backend", return_value="openvino"
            ):
                result = setup_openvino_test_config(temp_dir)
                expected = [
                    "keras_hub/src/models/gemma",
                    "keras_hub/src/tokenizers",
                ]
                self.assertEqual(result, expected)

    def test_contains_training_methods_with_training_code(self):
        """Test _contains_training_methods with file containing training
        methods."""
        training_code = """
        import keras

        def test_training_method():
            model = keras.Model()
            model.fit(x, y)  # Training method
            return model

        def test_other_method():
            model.compile(optimizer='adam')  # Training method
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
        """Test _contains_training_methods with nonexistent file."""
        result = _contains_training_methods(
            "/nonexistent/file.py", "test_method"
        )
        # According to the implementation, OSError returns True (conservative
        # approach)
        self.assertTrue(result)

    def test_should_auto_skip_training_test_non_python_file(self):
        """Test should_auto_skip_training_test with non-Python file."""

        class MockItem:
            def __init__(self, fspath):
                self.fspath = type("MockPath", (), {"basename": fspath})()
                self.name = "test_method"

        item = MockItem("test_file.txt")
        result = should_auto_skip_training_test(item)
        self.assertFalse(result)

    def test_should_auto_skip_training_test_non_openvino_backend(self):
        """Test should_auto_skip_training_test function behavior.

        Note: This function doesn't check the backend - it only analyzes
        code for training methods. Backend checking is done in
        get_openvino_skip_reason."""

        # Create a temporary file with simple test code (no training methods)
        simple_test_code = """
def test_method():
    # A simple test without training methods
    assert True
"""
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".py"
        ) as f:
            f.write(simple_test_code)
            temp_file = f.name

        try:

            class MockFspath:
                def __init__(self, path):
                    self.path = path
                    self.basename = os.path.basename(path)

                def __str__(self):
                    return self.path

            class MockItem:
                def __init__(self, fspath, name):
                    self.fspath = MockFspath(fspath)
                    self.name = name

            item = MockItem(temp_file, "test_method")

            # This function should return False for a simple test without
            # training methods, regardless of backend
            result = should_auto_skip_training_test(item)
            self.assertFalse(result)
        finally:
            os.unlink(temp_file)

    def test_should_auto_skip_training_test_with_training_methods(self):
        """Test should_auto_skip_training_test with training methods."""
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

            class MockFspath:
                def __init__(self, path):
                    self.path = path
                    self.basename = os.path.basename(path)

                def __str__(self):
                    return self.path

            class MockItem:
                def __init__(self, fspath, name):
                    self.fspath = MockFspath(fspath)
                    self.name = name

            item = MockItem(temp_file, "test_fit_method")

            with unittest.mock.patch(
                "keras.config.backend", return_value="openvino"
            ):
                result = should_auto_skip_training_test(item)
                self.assertTrue(result)
        finally:
            os.unlink(temp_file)

    def test_get_openvino_skip_reason_non_openvino_backend(self):
        """Test get_openvino_skip_reason with non-OpenVINO backend."""

        class MockItem:
            def __init__(self):
                self.name = "test_method"
                self.fspath = type("MockPath", (), {})()

        item = MockItem()
        with unittest.mock.patch(
            "keras.config.backend", return_value="tensorflow"
        ):
            result = get_openvino_skip_reason(item, [], True)
            self.assertIsNone(result)

    def test_get_openvino_skip_reason_specific_test_skip(self):
        """Test get_openvino_skip_reason with specific test methods that
        should be skipped."""

        class MockItem:
            def __init__(self, test_name):
                self.name = test_name
                self.fspath = type("MockPath", (), {})()
                setattr(self.fspath, "__str__", lambda: "test_file.py")

        with unittest.mock.patch(
            "keras.config.backend", return_value="openvino"
        ):
            # Define expected skip reasons matching the implementation
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
        """Test get_openvino_skip_reason with training methods."""
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

            class MockFspath:
                def __init__(self, path):
                    self.path = path
                    self.basename = os.path.basename(path)

                def __str__(self):
                    return self.path

            class MockItem:
                def __init__(self, fspath, name):
                    self.fspath = MockFspath(fspath)
                    self.name = name

            item = MockItem(temp_file, "test_training_method")

            with unittest.mock.patch(
                "keras.config.backend", return_value="openvino"
            ):
                result = get_openvino_skip_reason(item, [], True)
                self.assertEqual(result, "Training operations not supported")
        finally:
            os.unlink(temp_file)

    def test_get_openvino_skip_reason_whitelist_supported(self):
        """Test get_openvino_skip_reason with supported path in whitelist."""

        class MockFspath:
            def __init__(self, path):
                self.path = path
                self.basename = os.path.basename(path)

            def __str__(self):
                return self.path

        class MockItem:
            def __init__(self, fspath, name):
                self.fspath = MockFspath(fspath)
                self.name = name

        # Create a test path that should be supported
        test_path = "/some/path/keras_hub/src/models/gemma/gemma_test.py"
        supported_paths = ["keras_hub/src/models/gemma"]

        item = MockItem(test_path, "test_inference")

        with unittest.mock.patch(
            "keras.config.backend", return_value="openvino"
        ):
            result = get_openvino_skip_reason(item, supported_paths, False)
            self.assertIsNone(result)

    def test_get_openvino_skip_reason_whitelist_not_supported(self):
        """Test get_openvino_skip_reason with unsupported path not in
        whitelist."""

        class MockFspath:
            def __init__(self, path):
                self.path = path
                self.basename = os.path.basename(path)

            def __str__(self):
                return self.path

        class MockItem:
            def __init__(self, fspath, name):
                self.fspath = MockFspath(fspath)
                self.name = name

        # Create a test path that should NOT be supported
        test_path = "/some/path/keras_hub/src/models/gemma3/gemma3_test.py"
        supported_paths = ["keras_hub/src/models/gemma"]

        item = MockItem(test_path, "test_inference")

        with unittest.mock.patch(
            "keras.config.backend", return_value="openvino"
        ):
            result = get_openvino_skip_reason(item, supported_paths, False)
            self.assertEqual(result, "File/directory not in OpenVINO whitelist")

    def test_get_openvino_skip_reason_no_whitelist(self):
        """Test get_openvino_skip_reason with empty whitelist."""

        class MockFspath:
            def __init__(self, path):
                self.path = path
                self.basename = os.path.basename(path)

            def __str__(self):
                return self.path

        class MockItem:
            def __init__(self, fspath, name):
                self.fspath = MockFspath(fspath)
                self.name = name

        test_path = "/some/path/keras_hub/src/models/gemma/gemma_test.py"

        item = MockItem(test_path, "test_inference")

        with unittest.mock.patch(
            "keras.config.backend", return_value="openvino"
        ):
            result = get_openvino_skip_reason(item, [], False)
            self.assertIsNone(result)
