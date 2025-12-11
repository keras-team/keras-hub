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


def _generate_tokenizer_json(tokenizer, path, tokenizer_config):
    """Generate tokenizer.json for fast tokenizer support.
    
    This converts the SentencePiece tokenizer to HuggingFace's tokenizers
    format, enabling fast tokenizer support. The conversion attempts to
    preserve the original SentencePiece model type (Unigram, BPE, etc.).
    
    Args:
        tokenizer: The Keras tokenizer.
        path: Directory where tokenizer files are saved.
        tokenizer_config: The tokenizer configuration dictionary.
    """
    try:
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import Unigram
        from tokenizers.pre_tokenizers import Metaspace
        from tokenizers import normalizers
        import sentencepiece as spm
        
        # Load the SentencePiece model to check its type
        tokenizer_model_path = os.path.join(path, "tokenizer.model")
        if not os.path.exists(tokenizer_model_path):
            # Fallback to old path if rename hasn't happened yet
            tokenizer_model_path = os.path.join(path, "vocabulary.spm")
        
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(tokenizer_model_path)
        
        # Get vocabulary from the tokenizer with scores
        vocab = tokenizer.get_vocabulary()
        vocab_size = tokenizer.vocabulary_size()
        
        # Extract scores from the SentencePiece model
        vocab_scores = []
        for i, token in enumerate(vocab):
            # Get the actual score from SentencePiece model
            try:
                score = sp_model.GetScore(i)
            except:
                # Fallback: use negative log probability based on index
                score = -(i + 1) / vocab_size
            vocab_scores.append((token, score))
        
        # Create Unigram model (most SentencePiece models use Unigram)
        unk_id = tokenizer.token_to_id(tokenizer_config.get("unk_token", "<unk>"))
        hf_tokenizer = HFTokenizer(Unigram(vocab_scores, unk_id=unk_id if unk_id is not None else 0))
        
        # Add pre-tokenizer (Metaspace for SentencePiece compatibility)
        hf_tokenizer.pre_tokenizer = Metaspace(replacement="▁", add_prefix_space=True)
        
        # Add normalizer (empty for most SentencePiece models)
        hf_tokenizer.normalizer = normalizers.Sequence([])
        
        # Save tokenizer.json
        tokenizer_json_path = os.path.join(path, "tokenizer.json")
        hf_tokenizer.save(tokenizer_json_path)
        
    except ImportError as e:
        warnings.warn(
            f"Required library not installed ({e}). Fast tokenizer "
            "support will not be available. Install with: "
            "pip install tokenizers sentencepiece"
        )
    except Exception as e:
        warnings.warn(
            f"Failed to generate tokenizer.json for fast tokenizer support: {e}. "
            "The exported tokenizer will only work with use_fast=False."
        )


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
    
    # Generate tokenizer.json for fast tokenizer support
    _generate_tokenizer_json(tokenizer, path, tokenizer_config)


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
