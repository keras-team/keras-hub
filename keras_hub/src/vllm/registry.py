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
    
    When a model string starts with 'keras_hub:', vLLM will load KerasVLLMAdapter.
    """
    try:
        # Use vLLM's official extension/registration API
        ModelRegistry.register_model("KerasVLLMAdapter", KerasVLLMAdapter)
    except (ImportError, AttributeError, TypeError) as e:
        logging.warning("KerasVLLMAdapter registration skipped: %s", e)

