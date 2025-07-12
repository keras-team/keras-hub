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


def parameterize_inputs(inputs):
    if isinstance(inputs, (list, tuple)):
        return [parameterize_inputs(e) for e in inputs]
    elif isinstance(inputs, dict):
        return {k: parameterize_inputs(v) for k, v in inputs.items()}
    elif isinstance(inputs, np.ndarray):
        ov_type = OPENVINO_DTYPES[str(inputs.dtype)]
        ov_shape = list(inputs.shape)
        param = ov_opset.parameter(shape=ov_shape, dtype=ov_type)
        return ops.convert_to_tensor(param.output(0))
    elif isinstance(inputs, (int, np.integer)):
        param = ov_opset.parameter(shape=[], dtype=ov.Type.i32)
        return ops.convert_to_tensor(param.output(0))
    elif isinstance(inputs, (float, np.floating)):
        param = ov_opset.parameter(shape=[], dtype=ov.Type.f32)
        return ops.convert_to_tensor(param.output(0))
    else:
        raise TypeError(f"Unknown input type: {type(inputs)}")


def get_struct_outputs(inputs, stop_token_ids, fn):
    struct_params = parameterize_inputs(inputs)
    struct_outputs = fn(struct_params, stop_token_ids)
    return struct_params, struct_outputs


def get_outputs(inputs, struct_outputs, compile_ov_model):
    flatten_inputs = tree.flatten(inputs)
    for input in flatten_inputs:
        if ops.is_tensor(input):
            raise ValueError("inputs should be numpy arrays")
    outputs = compile_ov_model(flatten_inputs)
    outputs = unpack_singleton(
        tree.pack_sequence_as(struct_outputs, outputs.to_tuple())
    )
    return outputs
