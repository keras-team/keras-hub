import numpy as np
import openvino as ov
import openvino.runtime.opset14 as ov_opset
from keras import ops
from keras import tree

OPENVINO_DTYPES = {
    "float16": ov.Type.f16,
    "float32": ov.Type.f32,
    "float64": ov.Type.f64,
    "uint8": ov.Type.u8,
    "uint16": ov.Type.u16,
    "uint32": ov.Type.u32,
    "uint64": ov.Type.u64,
    "int8": ov.Type.i8,
    "int16": ov.Type.i16,
    "int32": ov.Type.i32,
    "int64": ov.Type.i64,
    "bfloat16": ov.Type.bf16,
    "bool": ov.Type.boolean,
    "float8_e4m3fn": ov.Type.f8e4m3,
    "float8_e5m2": ov.Type.f8e5m2,
    "string": ov.Type.string,
}


def unpack_singleton(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def parameterize_inputs(inputs, prefix=""):
    """
    Recursively converts input structures (dict, list, tuple, or scalars) into
    OpenVINO Parameter nodes, preserving structure and assigning friendly names.

    Args:
        inputs (Union[dict, list, tuple, np.ndarray, int, float]):
            Input data structure or value to parameterize.
        prefix (str): Prefix for naming OpenVINO parameter nodes.

    Returns:
        Structure of the same form as `inputs`, but with each input replaced
        by an OpenVINO-compatible tensor (converted parameter).

    Raises:
        TypeError: If the input type is not supported.
    """
    if isinstance(inputs, (list, tuple)):
        return [
            parameterize_inputs(e, f"{prefix}{i}") for i, e in enumerate(inputs)
        ]
    elif isinstance(inputs, dict):
        return {k: parameterize_inputs(v, k) for k, v in inputs.items()}
    elif isinstance(inputs, np.ndarray):
        ov_type = OPENVINO_DTYPES[str(inputs.dtype)]
        ov_shape = list(inputs.shape)
        param = ov_opset.parameter(shape=ov_shape, dtype=ov_type)
        param.set_friendly_name(prefix)
        return ops.convert_to_tensor(param.output(0))
    elif isinstance(inputs, (int, np.integer)):
        param = ov_opset.parameter(shape=[], dtype=ov.Type.i32)
        param.set_friendly_name(prefix)
        return ops.convert_to_tensor(param.output(0))
    elif isinstance(inputs, (float, np.floating)):
        param = ov_opset.parameter(shape=[], dtype=ov.Type.f32)
        param.set_friendly_name(prefix)
        return ops.convert_to_tensor(param.output(0))
    else:
        raise TypeError(f"Unknown input type: {type(inputs)}")


def get_struct_outputs(inputs, stop_token_ids, fn):
    """
    Prepares OpenVINO input parameters and calls the
    user-defined generation function.

    Args:
        inputs (dict or nested structure): Original input data
        stop_token_ids (Any): Stop token information passed to
        the model's generation step.
        fn (Callable): A function representing a single generation
        step that accepts parameterized inputs and returns structured outputs.

    Returns:
        Tuple: (parameterized_inputs, struct_outputs)
            - parameterized_inputs: OpenVINO parameter structure
            for model compilation.
            - struct_outputs: The output structure returned by
            the generation function.
    """
    struct_params = parameterize_inputs(inputs)
    struct_outputs = fn(struct_params, stop_token_ids)
    return struct_params, struct_outputs


def get_outputs(inputs, struct_outputs, compile_ov_model):
    """
    Executes the OpenVINO compiled model with the given
    inputs and reconstructs the output structure
    to match `struct_outputs`.

    Args:
        inputs (dict or nested structure): Original input data.
        struct_outputs (Any): The structure that defines
        how to reconstruct model outputs.
        compile_ov_model (Callable): The compiled OpenVINO
        model object with a `__call__` method.

    Returns:
        The model output reconstructed to
        match the structure of `struct_outputs`.

    Raises:
        ValueError: If any of the inputs are still tensors.
    """
    flatten_inputs = tree.flatten(inputs)
    for input in flatten_inputs:
        if ops.is_tensor(input):
            raise ValueError("inputs should be numpy arrays")
    outputs = compile_ov_model(flatten_inputs)
    outputs = unpack_singleton(
        tree.pack_sequence_as(struct_outputs, outputs.to_tuple())
    )
    return outputs
