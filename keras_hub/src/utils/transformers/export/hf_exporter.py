import gc
import json
import os
import shutil
import warnings

import keras

from keras_hub.src.utils.transformers.export.gemma import get_gemma_config
from keras_hub.src.utils.transformers.export.gemma import (
    get_gemma_tokenizer_config,
)
from keras_hub.src.utils.transformers.export.gemma import get_gemma_transform_fn
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

MODEL_TRANSFORMERS = {
    "GemmaBackbone": get_gemma_transform_fn,
    # Add for future models, e.g., "MistralBackbone": get_mistral_transform_fn
}


def get_native_tools():
    """
    Returns native tools for the active backend.
    """
    backend = keras.config.backend()

    if backend == "tensorflow":
        import tensorflow as tf
        from safetensors.tensorflow import save_file as tf_save

        def move_to_cpu(tensor):
            if not isinstance(tensor, (tf.Tensor, tf.Variable)):
                tensor = tf.convert_to_tensor(tensor)
            # Force CPU move immediately
            with tf.device("CPU:0"):
                return tf.identity(tensor)

        def get_size(tensor):
            nelem = tensor.shape.num_elements()
            if nelem is None:
                nelem = tf.reduce_prod(tf.shape(tensor)).numpy()
            return nelem * tensor.dtype.size

        def clear_mem():
            # Use GC only to avoid killing Tokenizer resources
            gc.collect()

        return move_to_cpu, tf_save, get_size, clear_mem

    elif backend == "torch":
        import torch
        from safetensors.torch import save_file as torch_save

        def move_to_cpu(tensor):
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            return tensor.detach().cpu()

        def get_size(tensor):
            return tensor.numel() * tensor.element_size()

        def clear_mem():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return move_to_cpu, torch_save, get_size, clear_mem

    elif backend == "jax":
        import jax
        from safetensors.flax import save_file as flax_save

        def move_to_cpu(tensor):
            return jax.device_put(tensor, jax.devices("cpu")[0])

        def get_size(tensor):
            return tensor.nbytes

        def clear_mem():
            jax.clear_caches()
            gc.collect()

        return move_to_cpu, flax_save, get_size, clear_mem

    else:
        raise ValueError(f"Unsupported Keras backend: {backend}")


def export_backbone(backbone, path, include_lm_head=False, max_shard_size=2.0):
    """Export the backbone model to HuggingFace format.

    Args:
        backbone: The Keras backbone model to convert.
        path: str. Path to save the exported model.
        include_lm_head: bool. If True, include lm_head weights if applicable.
        max_shard_size: float. Maximum size in GB for each shard.
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
    if model_type not in MODEL_TRANSFORMERS:
        raise ValueError(f"Transformations not implemented for {model_type}")

    # 1. Get Native Tools
    move_to_cpu, save_fn, get_size, clear_mem = get_native_tools()

    # Get config
    get_config_fn = MODEL_CONFIGS[model_type]
    hf_config = get_config_fn(backbone, include_lm_head=include_lm_head)
    # Save config
    os.makedirs(path, exist_ok=True)
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w") as f:
        json.dump(hf_config, f)
    # Get model-specific transform function
    get_transform_fn = MODEL_TRANSFORMERS[model_type]
    transform_fn = get_transform_fn(backbone)
    # Single pass: dynamic sharding based on actual tensor sizes,
    # processing one tensor at a time

    get_weights_fn = MODEL_EXPORTERS[model_type]
    weights_generator = get_weights_fn(
        backbone, include_lm_head=include_lm_head
    )
    shard_num = 1
    current_shard_dict = {}
    current_size_gb = 0.0
    weight_map = {}
    total_size_bytes = 0
    temp_shard_files = []
    current_temp_file = None

    for name, backend_tensor in weights_generator:
        # Move to CPU (Preserving dtype)
        cpu_tensor = move_to_cpu(backend_tensor)
        # Model-specific transform
        cpu_tensor = transform_fn(name, cpu_tensor)

        tensor_size_bytes = get_size(cpu_tensor)
        tensor_size_gb = tensor_size_bytes / (1024**3)
        total_size_bytes += tensor_size_bytes

        if (
            current_size_gb + tensor_size_gb > max_shard_size
            and current_shard_dict
        ):
            # Save current shard as temp
            current_temp_file = f"temp_shard_{shard_num}.safetensors"
            weights_path = os.path.join(path, current_temp_file)
            save_fn(current_shard_dict, weights_path, metadata={"format": "pt"})
            temp_shard_files.append(
                (current_temp_file, list(current_shard_dict.keys()))
            )
            del current_shard_dict
            current_shard_dict = {}
            current_size_gb = 0.0
            shard_num += 1
            clear_mem()

        current_shard_dict[name] = cpu_tensor
        current_size_gb += tensor_size_gb
        del cpu_tensor  # Explicitly del to aid GC after adding to shard

    # Save last shard
    if current_shard_dict:
        current_temp_file = f"temp_shard_{shard_num}.safetensors"
        weights_path = os.path.join(path, current_temp_file)
        save_fn(current_shard_dict, weights_path, metadata={"format": "pt"})
        temp_shard_files.append(
            (current_temp_file, list(current_shard_dict.keys()))
        )
        del current_shard_dict
        clear_mem()

    num_shards = shard_num
    # Rename temp files to final format and build weight_map
    for i, (temp_file, keys) in enumerate(temp_shard_files, 1):
        if num_shards == 1:
            final_file = "model.safetensors"
        else:
            final_file = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
        shutil.move(
            os.path.join(path, temp_file), os.path.join(path, final_file)
        )
        for key in keys:
            weight_map[key] = final_file
    # Save index
    index = {
        "metadata": {"total_size": total_size_bytes},
        "weight_map": weight_map,
    }
    index_path = os.path.join(path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f)


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


def export_to_safetensors(keras_model, path, max_shard_size=2.0):
    """Converts a Keras model to Hugging Face Transformers format.

    It does the following:
    - Exports the backbone (config and weights).
    - Exports the tokenizer assets.

    Args:
        keras_model: The Keras model to convert.
        path: str. Path of the directory to which the safetensors file,
          config and tokenizer will be saved.
        max_shard_size: float. Maximum size in GB for each shard during export.
    """
    if keras_model.preprocessor is not None:
        if keras_model.preprocessor.tokenizer is None:
            raise ValueError(
                "CausalLM preprocessor must have a tokenizer for export "
                "if attached."
            )
        export_tokenizer(keras_model.preprocessor.tokenizer, path)

    backbone = keras_model.backbone
    export_backbone(
        backbone, path, include_lm_head=True, max_shard_size=max_shard_size
    )
