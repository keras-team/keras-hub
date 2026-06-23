"""
Registry module for Keras Hub to vLLM Integration.

Hooks into vLLM's model loading mechanism so that `LLM(model="keras_hub:preset_name")`
is recognized and routed to the `KerasVLLMAdapter`.
"""
import logging
from vllm.model_executor.models import ModelRegistry

from keras_hub.src.vllm.adapter import KerasVLLMAdapter


def register_keras_hub_models() -> None:
    """Registers the keras_hub schema with vLLM's internal ModelRegistry.
    When `setup_vllm_model` is used to create a model directory, vLLM will natively 
    load the `KerasVLLMAdapter`.
    """
    try:
        from vllm.model_executor.models import ModelRegistry
        
        # Register the KerasVLLMAdapter with vLLM's internal registry
        ModelRegistry.register_model("KerasVLLMAdapter", KerasVLLMAdapter)
        
    except Exception as e:
        import logging
        logging.warning("KerasVLLMAdapter registration skipped: %s", e)


def setup_vllm_model(preset: str, dtype: str = "float16") -> str:
    """Creates a configuration directory for vLLM to load a Keras Hub preset.
    

    
    Args:
        preset: The Keras Hub preset name (e.g., "gemma_2b_en").
        dtype: The torch dtype to run inference with.
        
    Returns:
        The path to the temporary configuration directory to pass to `vllm.LLM`.
    """
    import tempfile
    import json
    import os
    
    temp_dir = tempfile.mkdtemp(prefix="keras_hub_vllm_")
    config_dict = {
        "architectures": ["KerasVLLMAdapter"],
        "_name_or_path": f"keras_hub:{preset}",
        "keras_hub_preset": preset,
        "torch_dtype": dtype,
    }
    
    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)
        
    return temp_dir
