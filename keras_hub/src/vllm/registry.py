"""
Registry module for Keras Hub to vLLM Integration.

Hooks into vLLM's model loading mechanism so that `LLM(model="keras_hub:preset_name")`
is recognized and routed to the `KerasVLLMAdapter`.
"""

import json
import logging
import os
import tempfile

from keras_hub.src.vllm.adapter import KerasVLLMAdapter


def _register_model_architecture() -> None:
    """Registers KerasVLLMAdapter with vLLM's internal model registry."""
    try:
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.register_model("KerasVLLMAdapter", KerasVLLMAdapter)
    except ImportError:
        logging.warning(
            "Skipping KerasVLLMAdapter registration. vLLM is not installed "
            "or the ModelRegistry module could not be imported."
        )


def _patch_tpu_model_loader() -> None:
    """Patches vLLM-TPU to treat KerasVLLMAdapter as a JAX-native architecture.

    If we don't do this, older versions of vLLM will fall back to tracing the PyTorch
    forward pass, which will cause JAX to capture Keras variables as large static constants.
    """
    try:
        from vllm.model_executor import model_loader

        if hasattr(model_loader, "JAX_NATIVE_ARCHITECTURES"):
            if "KerasVLLMAdapter" not in model_loader.JAX_NATIVE_ARCHITECTURES:
                model_loader.JAX_NATIVE_ARCHITECTURES = list(
                    model_loader.JAX_NATIVE_ARCHITECTURES
                ) + ["KerasVLLMAdapter"]
        if hasattr(model_loader, "_JAX_NATIVE_ARCHITECTURES"):
            if "KerasVLLMAdapter" not in model_loader._JAX_NATIVE_ARCHITECTURES:
                model_loader._JAX_NATIVE_ARCHITECTURES = list(
                    model_loader._JAX_NATIVE_ARCHITECTURES
                ) + ["KerasVLLMAdapter"]
    except (ImportError, AttributeError) as e:
        logging.debug("vLLM TPU model_loader patch skipped: %s", e)


def register_keras_hub_models() -> None:
    """Registers the keras_hub schema with vLLM's internal ModelRegistry.

    When `setup_vllm_model` is used to create a model directory, vLLM will
    natively load the `KerasVLLMAdapter`.
    """
    _register_model_architecture()
    _patch_tpu_model_loader()


def setup_vllm_model(preset: str, dtype: str = "float16") -> str:
    """Creates a configuration directory for vLLM to load a Keras Hub preset.

    Args:
        preset: The Keras Hub preset name (e.g., "gemma_2b_en").
        dtype: The torch dtype to run inference with.

    Returns:
        The path to the temporary configuration directory to pass to `vllm.LLM`.
    """
    temp_dir = tempfile.mkdtemp(prefix="keras_hub_vllm_")
    config_dict = {
        "architectures": ["KerasVLLMAdapter"],
        "_name_or_path": f"keras_hub:{preset}",
        "keras_hub_preset": preset,
        "torch_dtype": dtype,
        "model_type": "opt" if "opt" in preset.lower() else "gemma2",
        "vocab_size": 50272 if "opt" in preset.lower() else 256000,
    }

    with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    return temp_dir
