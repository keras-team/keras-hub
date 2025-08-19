import ast
import functools
from pathlib import Path

from keras import tree

from keras_hub.src.utils.keras_utils import print_msg

try:
    import openvino as ov
    import openvino.opset14 as ov_opset
    from openvino import Core

    core = Core()
except ImportError:
    ov = None
    ov_opset = None
    core = None


def load_openvino_supported_tools(config_file_path):
    """Load OpenVINO supported models from whitelist file.

    Args:
        config_file_path: Path to whitelist file.

    Returns:
        list: Supported model paths.
    """
    try:
        with open(config_file_path, "r") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except FileNotFoundError:
        return []


def setup_openvino_test_config(config_file_path):
    """Setup OpenVINO test configuration with whitelist approach.

    Args:
        config_file_path: Path to the config file directory.

    Returns:
        list: Supported paths (whitelist) for OpenVINO testing.
    """
    return load_openvino_supported_tools(
        Path(config_file_path) / "openvino_supported_tests.txt"
    )


@functools.lru_cache(maxsize=256)
def _contains_training_methods(file_path, test_name):
    """Check if a test function contains training methods.

    Args:
        file_path: Path to the test file.
        test_name: Name of the test function.

    Returns:
        bool: True if training methods found, False otherwise.
    """
    training_methods = {
        "fit",
        "fit_generator",
        "train_on_batch",
        "compile",
        "train_step",
        "train",
        "backward",
        "zero_grad",
        "step",
    }

    training_keywords = {
        "optimizer",
        "loss",
        "epochs",
        "batch_size",
        "learning_rate",
    }

    training_test_methods = {
        "run_layer_test",
        "run_training_step",
        "run_build_asserts",
        "run_task_test",
        "run_preprocessing_layer_test",
    }

    class TrainingMethodDetector(ast.NodeVisitor):
        def __init__(self):
            self.has_training_methods = False

        def visit_Call(self, node):
            if (
                hasattr(node.func, "attr")
                and node.func.attr in training_methods
            ):
                self.has_training_methods = True

            if (
                hasattr(node.func, "attr")
                and node.func.attr in training_test_methods
            ):
                self.has_training_methods = True

            if (
                hasattr(node.func, "value")
                and hasattr(node.func.value, "id")
                and node.func.value.id == "self"
                and hasattr(node.func, "attr")
                and node.func.attr in training_test_methods
            ):
                self.has_training_methods = True

            self.generic_visit(node)

        def visit_keyword(self, node):
            """Visit keyword arguments to detect training keywords."""
            if node.arg in training_keywords:
                self.has_training_methods = True
            self.generic_visit(node)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == test_name:
                detector = TrainingMethodDetector()
                detector.visit(node)
                return detector.has_training_methods
        return False
    except (OSError, SyntaxError):
        return True


def should_auto_skip_training_test(item):
    """Check if test should be auto-skipped for OpenVINO training ops.

    Args:
        item: Pytest test item.

    Returns:
        bool: True if should skip, False otherwise.
    """
    if not str(item.fspath).endswith(".py"):
        return False
    test_name = item.name.split("[")[0]
    return _contains_training_methods(str(item.fspath), test_name)


def get_openvino_skip_reason(item, supported_paths, auto_skip_training=True):
    """Whitelist-based OpenVINO test skip checker.

    Only tests files/directories in supported_paths, skips everything else.

    Args:
        item: Pytest test item.
        supported_paths: List of supported file/directory paths (whitelist).
        auto_skip_training: Whether to auto-skip training tests.

    Returns:
        str or None: Skip reason if should skip, None otherwise.
    """
    test_name = item.name.split("[")[0]
    test_path = str(item.fspath)

    # Priority 1: Skip specific problematic test methods
    SPECIFIC_SKIPPING_TESTS = {
        "test_backbone_basics": "Requires trainable backend",
        "test_score_loss": "Non-implemented roll operation",
        "test_layer_behaviors": "Requires trainable backend",
    }
    if test_name in SPECIFIC_SKIPPING_TESTS:
        return SPECIFIC_SKIPPING_TESTS[test_name]

    # Priority 2: Skip training operations (if enabled)
    if auto_skip_training and should_auto_skip_training_test(item):
        return "Training operations not supported"

    # Priority 3: Whitelist-based approach - only test supported paths
    if supported_paths:
        parts = test_path.replace("\\", "/").split("/")
        try:
            keras_hub_idx = parts.index("keras_hub")
            relative_test_path = "/".join(parts[keras_hub_idx:])
        except ValueError:
            relative_test_path = test_path  # fall back to absolute

        for supported_path in supported_paths:
            if (
                relative_test_path == supported_path
                or relative_test_path.startswith(supported_path + "/")
            ):
                return None  # in whitelist

        return "File/directory not in OpenVINO whitelist"

    return None


def get_device():
    """Detect and return the best available OpenVINO device.

    Returns:
        str: "GPU" if available, otherwise "CPU".
    """
    return "GPU" if "GPU" in core.available_devices else "CPU"


def compile_model(struct_params, struct_outputs, device, model_dtype):
    """Compile OpenVINO model with dynamic shapes and precision hints.

    Args:
        struct_params: Model parameters structure.
        struct_outputs: Model outputs structure.
        device: Target device ("GPU" or "CPU").
        model_dtype: Model precision ("f16" or "f32").

    Returns:
        Compiled OpenVINO model ready for inference.
    """
    flat_params = tree.flatten(struct_params)
    flat_outputs = tree.flatten(struct_outputs)
    parameters = [p.output.get_node() for p in flat_params]
    results = [ov_opset.result(r.output) for r in flat_outputs]
    ov_model = ov.Model(results=results, parameters=parameters)
    for ov_input in ov_model.inputs:
        rank = ov_input.get_partial_shape().rank.get_length()
        ov_input.get_node().set_partial_shape(ov.PartialShape([-1] * rank))
    ov_model.validate_nodes_and_infer_types()
    config = {"INFERENCE_PRECISION_HINT": model_dtype}
    return core.compile_model(ov_model, device, config)


def get_outputs(inputs, struct_outputs, compiled_ov_model, unpack_singleton):
    """Execute compiled OpenVINO model and return structured outputs.

    Args:
        inputs: Input tensors for inference.
        struct_outputs: Expected output structure.
        compiled_ov_model: Compiled OpenVINO model.
        unpack_singleton: Function to unpack singleton outputs.

    Returns:
        Structured model outputs matching expected format.
    """
    flatten_inputs = tree.flatten(inputs)
    raw = compiled_ov_model(flatten_inputs).to_tuple()
    packed = tree.pack_sequence_as(struct_outputs, raw)
    return unpack_singleton(packed)


def ov_infer(model, inputs, stop_token_ids, fn):
    """High-level OpenVINO inference with model reuse and compilation.

    This function manages OpenVINO model compilation and caching. It reuses
    existing compiled models when possible, or compiles new ones as needed.
    Handles device detection and automatic precision selection.

    Args:
        model: Keras model with OpenVINO backend support.
        inputs: Input tensors for inference.
        stop_token_ids: Token IDs that should stop generation.
        fn: Function to execute with the parameterized inputs.

    Returns:
        Model outputs from OpenVINO inference.
    """
    device = get_device()

    # Try to use existing compiled model for the same device
    if (
        getattr(model, "ov_compiled_model", None) is not None
        and getattr(model, "ov_device", None) is not None
        and device == model.ov_device
    ):
        try:
            return get_outputs(
                inputs,
                model.struct_outputs,
                model.ov_compiled_model,
                model._unpack_singleton,
            )
        except RuntimeError as e:
            print_msg(
                "WARNING: OpenVINO inference \033[1mFAILED\033[0m, "
                "recompiling model and trying again.\n" + str(e)
            )
            model.ov_compiled_model = None
            model.struct_outputs = None

    # Compile a new model
    struct_params = model._parameterize_data(inputs)
    model.struct_outputs = fn(struct_params, stop_token_ids)
    model.ov_device = device
    model_dtype = "f16" if model.dtype in ("float16", "bfloat16") else "f32"
    model.ov_compiled_model = compile_model(
        struct_params, model.struct_outputs, device, model_dtype
    )
    return get_outputs(
        inputs,
        model.struct_outputs,
        model.ov_compiled_model,
        model._unpack_singleton,
    )
