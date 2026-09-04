from keras import tree

from keras_hub.src.utils.keras_utils import print_msg

try:
    import openvino as ov
    import openvino.opset16 as ov_opset
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


def get_input_signature(inputs):
    """Summarize inputs as shapes and dtypes for cache invalidation.

    A compiled model is only valid for the shapes it was traced with, so this
    signature is what decides whether a cached model can be reused.

    Args:
        inputs: Input tensors, in any nested structure.

    Returns:
        tuple: A hashable summary of every leaf's shape and dtype.
    """
    signature = []
    for x in tree.flatten(inputs):
        shape = getattr(x, "shape", None)
        signature.append(
            (
                tuple(shape) if shape is not None else None,
                str(getattr(x, "dtype", None)),
            )
        )
    return tuple(signature)


def compile_model(struct_params, struct_outputs, device, model_dtype):
    """Compile an OpenVINO model with a precision hint.

    The graph keeps the shapes it was traced with, apart from the batch
    dimension that `_parameterize_data` already left dynamic. Relaxing the
    other dimensions here would let the model accept shapes it was never
    traced for, which fails at inference time.

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


def ov_infer(model, inputs, fn, static_args=(), cache_key=None):
    """High-level OpenVINO inference with model reuse and compilation.

    Compiles `fn` into an OpenVINO model the first time it is called, and
    reuses that model while the device, the input shapes and `cache_key` all
    stay the same. Anything that is baked into the graph as a constant rather
    than passed as an input must be reflected in `cache_key`, otherwise a
    later call would silently reuse a graph traced for different values.

    Args:
        model: Keras model with OpenVINO backend support.
        inputs: Input tensors to trace and run the model with.
        fn: Function to trace, called as `fn(struct_params, *static_args)`.
        static_args: Extra arguments passed to `fn` and baked in as constants.
        cache_key: Comparable summary of `static_args`. Defaults to
            `static_args`, which is only safe when those are plain values.

    Returns:
        Model outputs from OpenVINO inference.
    """
    device = get_device()
    if cache_key is None:
        cache_key = static_args
    signature = (device, get_input_signature(inputs), cache_key)

    # Reuse the compiled model only if it was traced for this exact signature.
    if (
        getattr(model, "ov_compiled_model", None) is not None
        and getattr(model, "ov_signature", None) == signature
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
            model.ov_signature = None

    # Compile a new model
    struct_params = model._parameterize_data(inputs)
    model.struct_outputs = fn(struct_params, *static_args)
    model.ov_device = device
    model_dtype = "f16" if model.dtype in ("float16", "bfloat16") else "f32"
    model.ov_compiled_model = compile_model(
        struct_params, model.struct_outputs, device, model_dtype
    )
    model.ov_signature = signature
    return get_outputs(
        inputs,
        model.struct_outputs,
        model.ov_compiled_model,
        model._unpack_singleton,
    )
