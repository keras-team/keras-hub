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
from keras_hub.src.utils.transformers.export.gemma3 import get_gemma3_config
from keras_hub.src.utils.transformers.export.gemma3 import (
    get_gemma3_image_converter_config,
)
from keras_hub.src.utils.transformers.export.gemma3 import (
    get_gemma3_processor_config,
)
from keras_hub.src.utils.transformers.export.gemma3 import (
    get_gemma3_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.gemma3 import (
    get_gemma3_weights_map,
)

MODEL_CONFIGS = {
    "GemmaBackbone": get_gemma_config,
    "Gemma3Backbone": get_gemma3_config,
    # Add for future models, e.g., "MistralBackbone": get_mistral_config
}

MODEL_EXPORTERS = {
    "GemmaBackbone": get_gemma_weights_map,
    "Gemma3Backbone": get_gemma3_weights_map,
    # Add for future models, e.g., "MistralBackbone": get_mistral_weights_map
}

MODEL_TOKENIZER_CONFIGS = {
    "GemmaTokenizer": get_gemma_tokenizer_config,
    "Gemma3Tokenizer": get_gemma3_tokenizer_config,
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

    # Rename files to match Hugging Face expectations

    # 1. SentencePiece Models (Gemma / Gemma 3)
    if tokenizer_type in ["GemmaTokenizer", "Gemma3Tokenizer"]:
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

    # 2. BPE Models (Qwen)
    elif tokenizer_type == "QwenTokenizer":
        vocab_json_path = os.path.join(path, "vocabulary.json")
        vocab_hf_path = os.path.join(path, "vocab.json")
        if os.path.exists(vocab_json_path):
            shutil.move(vocab_json_path, vocab_hf_path)
        else:
            warnings.warn(f"{vocab_json_path} not found.")

    # Generate tokenizer.json for models that need it
    if tokenizer_type == "Gemma3Tokenizer":
        try:
            from transformers import GemmaTokenizerFast

            hf_tokenizer = GemmaTokenizerFast.from_pretrained(path)
            hf_tokenizer.save_pretrained(path)
        except Exception as e:
            warnings.warn(
                f"Failed to generate tokenizer.json: {e}. "
                "Tokenizer may not handle special tokens correctly. "
                "Ensure 'transformers' is installed with sentencepiece support."
            )


def export_image_converter(backbone, path):
    """Export image converter config for vision models.

    Args:
        backbone: The Keras backbone model.
        path: str. Path to save the exported config.
    """
    model_type = backbone.__class__.__name__

    # Handle image converter config based on model type
    if model_type == "Gemma3Backbone":
        preprocessor_config = get_gemma3_image_converter_config(backbone)
        if preprocessor_config is not None:
            os.makedirs(path, exist_ok=True)
            preprocessor_config_path = os.path.join(
                path, "preprocessor_config.json"
            )
            with open(preprocessor_config_path, "w") as f:
                json.dump(preprocessor_config, f, indent=2)

    # Add future vision models here
    # elif model_type == "PaliGemmaBackbone":
    #     preprocessor_config = get_paligemma_image_converter_config(backbone)
    #     ...


def export_processor_config(backbone, path):
    """Export processor config for vision models.

    Args:
        backbone: The Keras backbone model.
        path: str. Path to save the exported config.
    """
    model_type = backbone.__class__.__name__

    # Handle processor config based on model type
    if model_type == "Gemma3Backbone":
        processor_config = get_gemma3_processor_config(backbone)
        if processor_config is not None:
            os.makedirs(path, exist_ok=True)
            processor_config_path = os.path.join(path, "processor_config.json")
            with open(processor_config_path, "w") as f:
                json.dump(processor_config, f, indent=2)
    # Add future vision models here


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

    # Export image converter and processor configs for vision models
    export_image_converter(backbone, path)
    export_processor_config(backbone, path)

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
