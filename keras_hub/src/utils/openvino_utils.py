from keras import tree

from keras_hub.src.utils.keras_utils import print_msg

try:
    import openvino as ov
    import openvino.opset14 as ov_opset
    from openvino import Core
except ImportError:
    ov = None
    ov_opset = None
    Core = None


_core = None


def get_core():
    """Get or create OpenVINO Core instance.

    Returns:
        openvino.Core: OpenVINO Core instance,
        or None if OpenVINO not available.
    """
    global _core
    if _core is None and Core is not None:
        _core = Core()
    return _core


def get_device():
    """Detect and return the best available OpenVINO device.

    Returns:
        str: "GPU" if available, otherwise "CPU".
    """
    core = get_core()
    if core is None:
        return "CPU"
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
    core = get_core()
    if core is None:
        raise RuntimeError("OpenVINO not available")
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
