"""
Integration module for vLLM and Keras Hub.

This module exposes the necessary adapters and hooks to allow vLLM to use Keras Hub
models natively as the backend for LLM generation.
"""

from .adapter import KerasVLLMAdapter
from .registry import KerasHubLLM
from .registry import register_keras_hub_models
from .registry import setup_vllm_model
from .tokenizer import KerasVLLMTokenizerAdapter

__all__ = [
    "KerasHubLLM",
    "KerasVLLMAdapter",
    "KerasVLLMTokenizerAdapter",
    "register_keras_hub_models",
    "setup_vllm_model",
]
