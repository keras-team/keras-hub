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
}

MODEL_EXPORTERS = {
    "GemmaBackbone": get_gemma_weights_map,
    "Gemma3Backbone": get_gemma3_weights_map,
    "QwenBackbone": get_qwen_weights_map,
}

MODEL_TOKENIZER_CONFIGS = {
    "GemmaTokenizer": get_gemma_tokenizer_config,
    "Gemma3Tokenizer": get_gemma3_tokenizer_config,
    "QwenTokenizer": get_qwen_tokenizer_config,
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
        # Lazy import to prevent crash on TF-only environments
        import torch
        from safetensors.torch import save_file

        weights_dict_torch = {}
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

            weights_dict_torch[k] = t

        # Handle Tied Weights
        if (
            "lm_head.weight" in weights_dict_torch
            and "model.embed_tokens.weight" in weights_dict_torch
        ):
            wte = weights_dict_torch["model.embed_tokens.weight"]
            lm = weights_dict_torch["lm_head.weight"]
            if wte.data_ptr() == lm.data_ptr():
                weights_dict_torch["lm_head.weight"] = lm.clone().contiguous()

        save_file(weights_dict_torch, weights_path, metadata={"format": "pt"})

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

    # Save tokenizer assets
    tokenizer.save_assets(path)

    # Export tokenizer config
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

    # 2. BPE Models (Qwen)
    elif tokenizer_type == "QwenTokenizer":
        vocab_json_path = os.path.join(path, "vocabulary.json")
        vocab_hf_path = os.path.join(path, "vocab.json")
        if os.path.exists(vocab_json_path):
            shutil.move(vocab_json_path, vocab_hf_path)
        else:
            warnings.warn(f"{vocab_json_path} not found.")


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
