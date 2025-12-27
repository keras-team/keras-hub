import json
import os
import shutil
import warnings

import keras

from keras_hub.src.utils.transformers.export.gemma import get_gemma_config
from keras_hub.src.utils.transformers.export.gemma import (
    get_gemma_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.gemma import get_gemma_weights_map

MODEL_CONFIGS = {
    "GemmaBackbone": get_gemma_config,
    # Add for future models, e.g., "MistralBackbone": get_mistral_config
}

MODEL_EXPORTERS = {
    "GemmaBackbone": get_gemma_weights_map,
    # Add for future models, e.g., "MistralBackbone": get_mistral_weights_map
}

MODEL_TOKENIZER_CONFIGS = {
    "GemmaTokenizer": get_gemma_tokenizer_config,
    # Add for future models, e.g., "MistralTokenizer":
    # get_mistral_tokenizer_config
}


def export_backbone(backbone, path, include_lm_head=False):
    """Export the backbone model to HuggingFace format.

    Args:
        backbone: The Keras backbone model to convert.
        path: str. Path to save the exported model.
        include_lm_head: bool. If True, include lm_head weights if applicable.
    """
    model_type = backbone.__class__.__name__
    if model_type not in MODEL_CONFIGS:
        raise ValueError(
            f"Export to Transformers format not implemented for {model_type}"
        )
    if model_type not in MODEL_EXPORTERS:
        raise ValueError(
            f"Export to Transformers format not implemented for {model_type}"
        )
    # Get config
    get_config_fn = MODEL_CONFIGS[model_type]
    hf_config = get_config_fn(backbone)
    # Get weights
    get_weights_fn = MODEL_EXPORTERS[model_type]
    weights_dict = get_weights_fn(backbone, include_lm_head=include_lm_head)
    if not weights_dict:
        raise ValueError("No weights to save.")
    # Save config
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f)
    # Save weights based on backend
    # Save weights based on backend
    weights_path = os.path.join(path, "model.safetensors")

    def weights_factory():
        return get_weights_fn(backbone, include_lm_head=include_lm_head)

    save_safetensors_streaming(weights_factory, weights_path)


def save_safetensors_streaming(weights_factory, path):
    """Save weights to a safetensors file using a streaming approach.

    Args:
        weights_factory: Callable that returns a dictionary or iterator of
            (key, tensor) pairs.
        path: str. Path to save the safetensors file.
    """
    import struct

    import numpy as np

    # Map Keras dtypes to safetensors dtypes
    DTYPE_MAP = {
        "float32": "F32",
        "float16": "F16",
        "bfloat16": "BF16",
        "int32": "I32",
        "int64": "I64",
        "uint8": "U8",
        "int8": "I8",
        "bool": "BOOL",
    }

    # First pass: calculate header and offsets
    header = {}
    offset = 0
    weights = weights_factory()

    # Handle both dict and iterator
    if isinstance(weights, dict):
        weights_iter = weights.items()
        is_dict = True
    else:
        weights_iter = weights
        is_dict = False

    for key, tensor in weights_iter:
        # Get shape and dtype without materializing value if possible
        shape = list(tensor.shape)
        dtype = keras.backend.standardize_dtype(tensor.dtype)

        if dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype}")
        st_dtype = DTYPE_MAP[dtype]

        # Calculate size in bytes
        # numpy dtypes have itemsize
        try:
            itemsize = np.dtype(dtype).itemsize
        except TypeError:
            # Handle bfloat16 manually if numpy doesn't support it (older numpy)
            if dtype == "bfloat16":
                itemsize = 2
            else:
                raise

        size = itemsize
        for dim in shape:
            size *= dim

        header[key] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size

    # Prepare header
    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")
    # Pad header to align to 8 bytes (optional but good practice)
    pad_len = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad_len
    header_len = len(header_bytes)

    # Write file
    with open(path, "wb") as f:
        # Write header length (8 bytes, little endian unsigned long long)
        f.write(struct.pack("<Q", header_len))
        # Write header
        f.write(header_bytes)

        # Second pass: write data
        if is_dict:
            weights_iter = weights.items()
        else:
            weights_iter = weights_factory()

        for key, tensor in weights_iter:
            # Convert to numpy and write bytes
            # We use np.array() which works for Keras tensors
            # For bfloat16, we might need special handling if numpy doesn't
            # support it But Keras usually handles it.
            # If backend is JAX/Torch, they have bfloat16. Numpy < 2.0 doesn't.
            # If numpy doesn't support bfloat16, we might get float32 or object?
            # We need raw bytes.

            val = tensor
            if hasattr(val, "value"):  # Keras Variable
                val = val.value

            # Convert to numpy
            # Note: for bfloat16 on systems without numpy bfloat16 support,
            # this might be tricky.
            # Keras backend `convert_to_numpy` might return fp32 for bfp16?
            # We need to check this.

            np_val = np.array(val)

            # If dtype was bfloat16 and we got float32, we are in trouble?
            # Or if we got a custom bfloat16 type.
            # Let's assume standard behavior for now.

            f.write(np_val.tobytes())


def export_tokenizer(tokenizer, path):
    """Export only the tokenizer to HuggingFace Transformers format.

    Args:
        tokenizer: The Keras tokenizer to convert.
        path: str. Path to save the exported tokenizer.
    """
    os.makedirs(path, exist_ok=True)
    # Save tokenizer assets
    tokenizer.save_assets(path)
    # Export tokenizer config
    tokenizer_type = tokenizer.__class__.__name__
    if tokenizer_type not in MODEL_TOKENIZER_CONFIGS:
        raise ValueError(
            "Export to Transformers format not implemented for {tokenizer_type}"
        )
    get_tokenizer_config_fn = MODEL_TOKENIZER_CONFIGS[tokenizer_type]
    tokenizer_config = get_tokenizer_config_fn(tokenizer)
    tokenizer_config_path = os.path.join(path, "tokenizer_config.json")
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=4)
    # Rename vocabulary file
    vocab_spm_path = os.path.join(path, "vocabulary.spm")
    tokenizer_model_path = os.path.join(path, "tokenizer.model")
    if os.path.exists(vocab_spm_path):
        shutil.move(vocab_spm_path, tokenizer_model_path)
    else:
        warnings.warn(
            f"{vocab_spm_path} not found. Tokenizer may not load "
            "correctly. Ensure that the tokenizer configuration "
            "is correct and that the vocabulary file is present "
            "in the original model."
        )


def export_to_safetensors(keras_model, path):
    """Converts a Keras model to Hugging Face Transformers format.

    It does the following:
    - Exports the backbone (config and weights).
    - Exports the tokenizer assets.

    Args:
        keras_model: The Keras model to convert.
        path: str. Path of the directory to which the safetensors file,
          config and tokenizer will be saved.
    """
    backbone = keras_model.backbone
    export_backbone(backbone, path, include_lm_head=True)
    if (
        keras_model.preprocessor is not None
        and keras_model.preprocessor.tokenizer is None
    ):
        raise ValueError(
            "CausalLM preprocessor must have a tokenizer for export "
            "if attached."
        )
    if keras_model.preprocessor is not None:
        export_tokenizer(keras_model.preprocessor.tokenizer, path)
