"""
Integration module for vLLM and Keras Hub.

This module exposes the necessary adapters and hooks to allow vLLM to use Keras Hub
models natively as the backend for LLM generation.
"""

from .adapter import KerasVLLMAdapter
from .tokenizer import KerasVLLMTokenizerAdapter
from .registry import register_keras_hub_models

__all__ = [
    "KerasVLLMAdapter",
    "KerasVLLMTokenizerAdapter",
    "register_keras_hub_models",
]
