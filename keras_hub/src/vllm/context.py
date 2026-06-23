"""
Thread-local context management for passing vLLM metadata to Keras layers.
"""

import threading
from typing import Any, Optional


class VLLMContext(threading.local):
    """Thread-local context for passing serving metadata to Keras layers.

    Attributes:
        block_tables: The block tables tensor for paged attention.
        slot_mapping: The slot mapping tensor for paged attention.
        attention_metadata: The full attention metadata object from vLLM.
        paged_attention_func: The compiled hardware-specific paged attention kernel.
        active: Boolean indicating if the context is currently active.
    """

    def __init__(self) -> None:
        """Initializes an empty inactive context."""
        super().__init__()
        self.block_tables: Optional[Any] = None
        self.slot_mapping: Optional[Any] = None
        self.attention_metadata: Optional[Any] = None
        self.paged_attention_func: Optional[Any] = None
        self.active: bool = False


_vllm_context = VLLMContext()


def set_vllm_context(
    block_tables: Any,
    slot_mapping: Any,
    attention_metadata: Optional[Any] = None,
    paged_attention_func: Optional[Any] = None,
) -> None:
    """Sets the thread-local vLLM context parameters.

    Args:
        block_tables: Array representing memory blocks for key/value caching.
        slot_mapping: Array mapping sequence tokens to cache slots.
        attention_metadata: Additional hardware/framework specific metadata.
        paged_attention_func: The function to use for paged attention.
    """
    _vllm_context.block_tables = block_tables
    _vllm_context.slot_mapping = slot_mapping
    _vllm_context.attention_metadata = attention_metadata
    _vllm_context.paged_attention_func = paged_attention_func
    _vllm_context.active = True


def clear_vllm_context() -> None:
    """Clears the thread-local vLLM context."""
    _vllm_context.block_tables = None
    _vllm_context.slot_mapping = None
    _vllm_context.attention_metadata = None
    _vllm_context.paged_attention_func = None
    _vllm_context.active = False


def get_vllm_context() -> Optional[VLLMContext]:
    """Retrieves the active thread-local vLLM context.

    Returns:
        The `VLLMContext` instance if active, otherwise `None`.
    """
    return _vllm_context if getattr(_vllm_context, "active", False) else None
