import json
import os
import shutil
import warnings

import keras

from keras_hub.src.utils.transformers.export.gemma import get_gemma_config
from keras_hub.src.utils.transformers.export.gemma import get_gemma_weights_map

MODEL_CONFIGS = {
    "GemmaBackbone": get_gemma_config,
    # Add future models here, e.g., "LlamaBackbone": get_llama_config,
}

MODEL_EXPORTERS = {
    "GemmaBackbone": get_gemma_weights_map,
    # Add future models here, e.g., "LlamaBackbone": get_llama_weights_map,
}


def export_to_safetensors(keras_model, path, verbose=True):
    """Converts a Keras model to Hugging Face safetensor format.
    It does the following:
    - Exports the backbone (config and weights).
    - Exports the tokenizer assets.
    Args:
        keras_model: The Keras model to convert.
        path: str. Path of the directory to which the safetensors file,
          config and tokenizer will be saved.
        verbose: bool. If True, print success messages (default: True).
    """
    backbone = keras_model.backbone
    export_backbone(backbone, path, verbose=verbose)
    tokenizer = keras_model.preprocessor.tokenizer
    export_tokenizer(tokenizer, path, verbose=verbose)
    if verbose:
        print(f"Model exported successfully to {path}")


def export_backbone(backbone, path, verbose=True):
    """Export only the backbone model to HuggingFace format.
    Args:
        backbone: The Keras backbone model to convert.
        path: str. Path to save the exported model.
        verbose: bool. If True, print success messages (default: True).
    """
    backend = keras.config.backend()
    model_type = backbone.__class__.__name__
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Config not implemented for {model_type}")
    if model_type not in MODEL_EXPORTERS:
        raise ValueError(f"Exporter not implemented for {model_type}")
    # Get config
    get_config_fn = MODEL_CONFIGS[model_type]
    hf_config = get_config_fn(backbone)
    # Get weights
    get_weights_fn = MODEL_EXPORTERS[model_type]
    weights_dict = get_weights_fn(backbone)
    if not weights_dict:
        raise ValueError("No weights to save.")
    # Save config
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f)
    # Save weights based on backend
    weights_path = os.path.join(path, "model.safetensors")
    if backend == "torch":
        from safetensors.torch import save_file

        weights_dict_contiguous = {
            k: v.value.contiguous() if hasattr(v, "value") else v.contiguous()
            for k, v in weights_dict.items()
        }
        save_file(
            weights_dict_contiguous, weights_path, metadata={"format": "pt"}
        )
    elif backend == "tensorflow":
        from safetensors.tensorflow import save_file

        save_file(weights_dict, weights_path, metadata={"format": "pt"})
    elif backend == "jax":
        from safetensors.flax import save_file

        save_file(weights_dict, weights_path, metadata={"format": "pt"})
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    if verbose:
        print(f"Backbone exported successfully to {path}")


def export_tokenizer(tokenizer, path, verbose=True):
    """Export only the tokenizer to HuggingFace format.
    Args:
        tokenizer: The Keras tokenizer to convert.
        path: str. Path to save the exported tokenizer.
        verbose: bool. If True, print success messages (default: True).
    """
    os.makedirs(path, exist_ok=True)
    # Save tokenizer assets
    tokenizer.save_assets(path)
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
    if verbose:
        print(f"Tokenizer exported successfully to {path}")
