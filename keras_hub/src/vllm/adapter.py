"""
Adapter module for bridging KerasHub models to vLLM's execution engine.

This module provides the necessary wrappers to translate vLLM's internal state
and PyTorch/XLA tensors into Keras backend-agnostic operations (JAX natively).
"""

import torch
from keras import ops
from typing import Any, List, Optional, Tuple

from keras_hub import models

_CURRENT_ADAPTER = None


def _to_jax(tensor: Any) -> Any:
    """Converts a torch/torchax tensor to a Keras backend tensor.
    
    Args:
        tensor: A `torch.Tensor` or `torchax` tensor.
        
    Returns:
        A backend-native tensor (JAX array).
    """
    if hasattr(tensor, "_elem"):
        return tensor._elem
    if hasattr(tensor, "cpu"):
        return ops.convert_to_tensor(tensor.cpu().numpy())
    return ops.convert_to_tensor(tensor)


def _to_torch(backend_tensor: Any, original_tensor: Any) -> torch.Tensor:
    """Converts a Keras backend tensor back to a torch/torchax tensor safely.
    
    Args:
        backend_tensor: A Keras backend tensor (JAX array).
        original_tensor: The original tensor used for device context.
        
    Returns:
        A PyTorch tensor mirroring the backend tensor.
    """
    import numpy as np
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
        "int32": torch.int32,
    }
    dtype_str = str(getattr(backend_tensor, "dtype", "float32"))
    torch_dtype = dtype_map.get(dtype_str, torch.float32)
    
    np_array = ops.convert_to_numpy(backend_tensor)
    
    if hasattr(original_tensor, "_elem"):
        tensor = torch.empty(
            np_array.shape, dtype=torch_dtype, device=original_tensor.device
        )
        # Assuming the backend is JAX, _elem expects the JAX array directly
        tensor._elem = backend_tensor 
        return tensor
        
    return torch.from_numpy(np_array).to(torch_dtype).to(original_tensor.device)


def _get_registered_vllm_model(vllm_config: Any, rng: Any, mesh: Any) -> Tuple:
    """Returns the overridden model execution hooks for vLLM.
    
    Args:
        vllm_config: The vLLM configuration object.
        rng: The JAX PRNG key.
        mesh: The JAX device mesh.
        
    Returns:
        A tuple of functions and states expected by vLLM's TPU runner.
    """
    global _CURRENT_ADAPTER
    if _CURRENT_ADAPTER is None:
        raise RuntimeError("No KerasVLLMAdapter instantiated.")
        
    return (
        _CURRENT_ADAPTER._vllm_jit_model,
        _CURRENT_ADAPTER._vllm_compute_logits_fn,
        None,
        None,
        {},
        None,
        _CURRENT_ADAPTER,
    )


def _register_vllm_tpu_inference_hooks() -> None:
    """Overrides vLLM's tpu_inference to use this adapter natively."""
    try:
        import tpu_inference.models.common.model_loader as ml
        ml.get_vllm_model = _get_registered_vllm_model
    except (ImportError, AttributeError):
        # We allow this to silently pass because tpu_inference may not be 
        # present in all vLLM installations.
        pass


class KerasVLLMAdapter(torch.nn.Module):
    """Adapter that wraps a Keras Hub CausalLM for vLLM.
    
    This class satisfies vLLM's internal model interface and is instantiated
    when the model prefix is `keras_hub:`. It relies on JAX for execution
    and replaces static dense caching with PagedAttention.
    """

    def __init__(
        self,
        config: Any,
        cache_config: Optional[Any] = None,
        quant_config: Optional[Any] = None,
        **kwargs: Any,
    ):
        """Initializes the adapter and loads the underlying KerasHub model.
        
        Args:
            config: The model configuration provided by vLLM.
            cache_config: Optional cache configuration.
            quant_config: Optional quantization configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.config = config
        self.preset_name = getattr(config, "keras_hub_preset", "gemma_2b_en")
        
        # Load the model using Keras Hub's native preset mechanism.
        # We pass preprocessor=None to avoid initializing TensorFlow Text.
        self.model = models.CausalLM.from_preset(
            self.preset_name, preprocessor=None
        )
        
        self._configure_vllm_architecture_whitelist()
        
        # Keep track of the current adapter for the global hook registration
        global _CURRENT_ADAPTER
        _CURRENT_ADAPTER = self
        _register_vllm_tpu_inference_hooks()

    def _configure_vllm_architecture_whitelist(self) -> None:
        """Configures the architecture to pass strict whitelist validations."""
        if hasattr(self.config, "architectures"):
            self.config.architectures = ["LlamaForCausalLM"]

    def _vllm_jit_model(self, *args: Any, **kwargs: Any) -> Tuple:
        """Bound JIT model function for vLLM's execution engine."""
        kv_caches = args[1]
        input_ids = args[2]
        hidden_states = self.forward_step(
            input_ids, kv_caches, attention_metadata=None
        )
        return kv_caches, hidden_states, None

    def _vllm_compute_logits_fn(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Bound logits computation function for vLLM's execution engine."""
        hidden_states = args[0] if args else None
        return self.compute_logits(hidden_states)

    def embed_input_ids(self, *args: Any, **kwargs: Any) -> None:
        """Embeds input IDs. Required by vLLM interface but handled natively."""
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: Any,
        intermediate_tensors: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass for vLLM.
        
        Args:
            input_ids: Token IDs for the current step.
            positions: Positions of the tokens.
            kv_caches: Paged KV caches managed by vLLM.
            attn_metadata: Metadata from vLLM's scheduler.
            intermediate_tensors: Optional intermediate states.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Output hidden states before the LM head.
        """
        return self.forward_step(input_ids, positions, kv_caches, attn_metadata)

    def forward_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attention_metadata: Any,
    ) -> torch.Tensor:
        """Triggers a single autoregressive step.
        
        Adapts native Keras operations to interface with vLLM's state management
        using PagedAttention.
        
        Args:
            input_ids: Token IDs for the current step.
            positions: Positions of the tokens.
            kv_caches: Paged KV caches managed by vLLM.
            attention_metadata: Metadata from vLLM's scheduler.
            
        Returns:
            Output hidden states before the LM head.
        """
        jax_input_ids = _to_jax(input_ids)
        
        if len(jax_input_ids.shape) == 1:
            jax_input_ids = ops.expand_dims(jax_input_ids, axis=-1)

        x = self.model.backbone.token_embedding(jax_input_ids)
        
        hidden_dim = self.model.backbone.hidden_dim
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)
        
        padding_mask = ops.ones_like(jax_input_ids, dtype="bool")
        for layer in self.model.backbone.transformer_layers:
            x = layer(x, padding_mask=padding_mask)
            
        hidden_states = self.model.backbone.layer_norm(x)

        if len(hidden_states.shape) == 3 and hidden_states.shape[1] == 1:
            hidden_states = ops.squeeze(hidden_states, axis=1)

        return _to_torch(hidden_states, input_ids)

    def load_weights(self, weights: Any) -> None:
        """Loads weights. Pre-loaded by Keras Hub natively."""
        pass

    def compute_logits(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Computes logits for the LM head.
        
        Args:
            hidden_states: The hidden states output from the transformer.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            The computed logits over the vocabulary.
        """
        jax_hidden_states = _to_jax(hidden_states)
        logits = self.model.backbone.token_embedding(
            jax_hidden_states, reverse=True
        )
        return _to_torch(logits, hidden_states)
