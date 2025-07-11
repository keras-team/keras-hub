import json
import os
import shutil
import warnings

import keras

from .gemma import get_gemma_config
from .gemma import get_gemma_weights_map

MODEL_CONFIGS = {
    "GemmaBackbone": get_gemma_config,
    # Add future models here, e.g., "LlamaBackbone": get_llama_config,
}

MODEL_EXPORTERS = {
    "GemmaBackbone": get_gemma_weights_map,
    # Add future models here, e.g., "LlamaBackbone": get_llama_weights_map,
}


def export_to_safetensors(keras_model, path):
    """This function converts a Keras model to Hugging Face format by:
    - Extracting and mapping weights from the Keras backbone to safetensors.
    - Saving the configuration as 'config.json'.
    - Saving weights in 'model.safetensors'.
    - Saving tokenizer assets.
    Args:
        keras_model: The Keras model to convert.
        path: str. Path of the directory to which the safetensors file,
          config and tokenizer will be saved.
    """
    backend = keras.config.backend()
    backbone = keras_model.backbone
    model_type = backbone.__class__.__name__

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Config not implemented for {model_type}")

    if model_type not in MODEL_EXPORTERS:
        raise ValueError(f"Exporter not implemented for {model_type}")

    get_config_fn = MODEL_CONFIGS[model_type]
    hf_config = get_config_fn(backbone)

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
        try:
            from safetensors.torch import save_file

            weights_dict_contiguous = {
                k: v.value.contiguous()
                if hasattr(v, "value")
                else v.contiguous()
                for k, v in weights_dict.items()
            }
            save_file(
                weights_dict_contiguous, weights_path, metadata={"format": "pt"}
            )
        except ImportError:
            raise ImportError("Install `safetensors.torch` for Torch backend.")
    elif backend == "tensorflow":
        try:
            from safetensors.tensorflow import save_file

            save_file(weights_dict, weights_path, metadata={"format": "pt"})
        except ImportError:
            raise ImportError(
                "Install `safetensors.tensorflow` for TensorFlow backend."
            )
    elif backend == "jax":
        try:
            from safetensors.flax import save_file

            weights_dict_contiguous = {k: v for k, v in weights_dict.items()}
            save_file(
                weights_dict_contiguous, weights_path, metadata={"format": "pt"}
            )
        except ImportError:
            raise ImportError("Install `safetensors.flax` for JAX backend.")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Save tokenizer assets
    keras_model.preprocessor.tokenizer.save_assets(path)

    # Rename vocabulary file
    vocab_spm_path = os.path.join(path, "vocabulary.spm")
    tokenizer_model_path = os.path.join(path, "tokenizer.model")
    if os.path.exists(vocab_spm_path):
        shutil.move(vocab_spm_path, tokenizer_model_path)
    else:
        warnings.warn(
            f"{vocab_spm_path} not found.Tokenizer may not load "
            "correctly. Ensure that the tokenizer configuration "
            "is correct and that the vocabulary file is present "
            "in the original model."
        )
