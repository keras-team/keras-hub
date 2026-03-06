import json
import os
import shutil
import warnings

import keras

# --- Gemma Utils ---
from keras_hub.src.utils.transformers.export.gemma import get_gemma_config
from keras_hub.src.utils.transformers.export.gemma import (
    get_gemma_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.gemma import get_gemma_weights_map

# --- Gemma 3 Utils ---
from keras_hub.src.utils.transformers.export.gemma3 import get_gemma3_config
from keras_hub.src.utils.transformers.export.gemma3 import (
    get_gemma3_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.gemma3 import (
    get_gemma3_weights_map,
)

# --- GPT2 Utils ---
from keras_hub.src.utils.transformers.export.gpt2 import get_gpt2_config
from keras_hub.src.utils.transformers.export.gpt2 import (
    get_gpt2_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.gpt2 import get_gpt2_weights_map

# --- Qwen Utils ---
from keras_hub.src.utils.transformers.export.qwen import get_qwen_config
from keras_hub.src.utils.transformers.export.qwen import (
    get_qwen_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.qwen import get_qwen_weights_map

MODEL_CONFIGS = {
    "GemmaBackbone": get_gemma_config,
    "Gemma3Backbone": get_gemma3_config,
    "QwenBackbone": get_qwen_config,
    "GPT2Backbone": get_gpt2_config,
}

MODEL_EXPORTERS = {
    "GemmaBackbone": get_gemma_weights_map,
    "Gemma3Backbone": get_gemma3_weights_map,
    "QwenBackbone": get_qwen_weights_map,
    "GPT2Backbone": get_gpt2_weights_map,
}

MODEL_TOKENIZER_CONFIGS = {
    "GemmaTokenizer": get_gemma_tokenizer_config,
    "Gemma3Tokenizer": get_gemma3_tokenizer_config,
    "QwenTokenizer": get_qwen_tokenizer_config,
    "GPT2Tokenizer": get_gpt2_tokenizer_config,
}


def export_backbone(backbone, path, include_lm_head=False):
    """Export the backbone model to HuggingFace format.

    Args:
        backbone: The Keras backbone model to convert.
        path: str. Path to save the exported model.
        include_lm_head: bool. If True, include lm_head weights if applicable.
    """
    backend = keras.config.backend()
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

    config_to_save = hf_config
    if hasattr(hf_config, "to_dict"):
        config_to_save = hf_config.to_dict()

    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    # Save weights based on backend
    weights_path = os.path.join(path, "model.safetensors")
    if backend == "torch":
        import struct

        import torch
        from safetensors.torch import _SIZE

        _DTYPE_MAP = {
            "float32": "F32",
            "bfloat16": "BF16",
            "float16": "F16",
            "int64": "I64",
            "int32": "I32",
            "int16": "I16",
            "int8": "I8",
            "uint8": "U8",
            "bool": "BOOL",
            "float64": "F64",
        }

        # Pass 1: generate metadata
        header = {"__metadata__": {"format": "pt"}}
        offset = 0
        for k, v in weights_dict.items():
            tensor = v.value if hasattr(v, "value") else v

            if isinstance(tensor, torch.Tensor):
                t = tensor.detach().to("cpu")
            elif hasattr(tensor, "numpy"):
                t = torch.tensor(tensor.numpy())
            elif hasattr(tensor, "__array__"):
                t = torch.tensor(tensor)
            else:
                t = tensor

            dtype_str = str(t.dtype).split(".")[-1]
            dtype_mapped = _DTYPE_MAP.get(dtype_str, "F32")

            shape = list(t.shape)
            byte_size = t.nelement() * _SIZE[t.dtype]

            header[k] = {
                "dtype": dtype_mapped,
                "shape": shape,
                "data_offsets": [offset, offset + byte_size],
            }
            offset += byte_size

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad_len = (8 - len(header_json) % 8) % 8
        header_json += b" " * pad_len
        header_len = len(header_json)

        # Pass 2: write data streamingly
        # Handles model writing one tensor at a time, avoiding OOMs
        with open(weights_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_json)

            for k, v in weights_dict.items():
                tensor = v.value if hasattr(v, "value") else v

                if isinstance(tensor, torch.Tensor):
                    t = tensor.detach().to("cpu")
                elif hasattr(tensor, "numpy"):
                    t = torch.tensor(tensor.numpy())
                elif hasattr(tensor, "__array__"):
                    t = torch.tensor(tensor)
                else:
                    t = tensor

                if hasattr(t, "contiguous"):
                    t = t.contiguous()

                b = t.view(torch.uint8).numpy().tobytes()
                f.write(b)

    elif backend == "tensorflow":
        from safetensors.tensorflow import save_file

        save_file(weights_dict, weights_path, metadata={"format": "pt"})
    elif backend == "jax":
        from safetensors.flax import save_file

        save_file(weights_dict, weights_path, metadata={"format": "pt"})
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def export_tokenizer(tokenizer, path):
    """Export only the tokenizer to HuggingFace Transformers format.

    Args:
        tokenizer: The Keras tokenizer to convert.
        path: str. Path to save the exported tokenizer.
    """
    os.makedirs(path, exist_ok=True)
    tokenizer.save_assets(path)

    tokenizer_type = tokenizer.__class__.__name__
    if tokenizer_type not in MODEL_TOKENIZER_CONFIGS:
        raise ValueError(
            f"Export to Transformer format not implemented for {tokenizer_type}"
        )

    get_tokenizer_config_fn = MODEL_TOKENIZER_CONFIGS[tokenizer_type]
    tokenizer_config = get_tokenizer_config_fn(tokenizer)
    tokenizer_config_path = os.path.join(path, "tokenizer_config.json")
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=4)

    # Rename files to match Hugging Face expectations

    # 1. SentencePiece Models (Gemma / Gemma 3)
    if tokenizer_type in ["GemmaTokenizer", "Gemma3Tokenizer"]:
        vocab_spm_path = os.path.join(path, "vocabulary.spm")
        tokenizer_model_path = os.path.join(path, "tokenizer.model")
        if os.path.exists(vocab_spm_path):
            shutil.move(vocab_spm_path, tokenizer_model_path)
        else:
            warnings.warn(f"{vocab_spm_path} not found.")

    # 2. BPE Models (Qwen / GPT-2)
    elif tokenizer_type in ["QwenTokenizer", "GPT2Tokenizer"]:
        vocab_json_path = os.path.join(path, "vocabulary.json")
        vocab_hf_path = os.path.join(path, "vocab.json")
        if os.path.exists(vocab_json_path):
            shutil.move(vocab_json_path, vocab_hf_path)
        else:
            warnings.warn(
                f"{vocab_json_path} not found.Tokenizer may not load correctly."
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
        raise ValueError("CausalLM preprocessor must have a tokenizer.")
    if keras_model.preprocessor is not None:
        export_tokenizer(keras_model.preprocessor.tokenizer, path)
