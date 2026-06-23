"""
Adapter module for bridging KerasHub models to vLLM's execution engine.

This module provides the necessary wrappers to translate vLLM's internal state
and PyTorch/XLA tensors into Keras backend-agnostic operations (JAX natively).
"""

import numpy as np
import torch
from keras import ops
from typing import Any, List, Optional, Tuple

from keras_hub import models

try:
    import tpu_inference.models.common.model_loader as ml
except ImportError:
    ml = None

_CURRENT_ADAPTER = None


def _to_jax(tensor: Any) -> Any:
    """Converts a torch/torchax tensor to a Keras backend tensor via DLPack.
    
    Args:
        tensor: A `torch.Tensor` or `torchax` tensor.
        
    Returns:
        A backend-native tensor (JAX array).
    """
    if hasattr(tensor, "_elem"):
        return tensor._elem
    import jax
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous()))


def _to_torch(backend_tensor: Any, original_tensor: Any) -> torch.Tensor:
    """Converts a Keras backend tensor back to a torch tensor via DLPack safely.
    
    Args:
        backend_tensor: A Keras backend tensor (JAX array).
        original_tensor: The original tensor used for device context.
        
    Returns:
        A PyTorch tensor mirroring the backend tensor.
    """
    if hasattr(original_tensor, "_elem"):
        # We assume the backend_tensor is already a JAX array compatible with _elem
        return backend_tensor

    import jax
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(backend_tensor))

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
        """Initializes the adapter and loads the underlying KerasHub model."""
        super().__init__()
        self.config = config
        self.preset_name = getattr(config, "keras_hub_preset", None)
        if self.preset_name is None:
            raise ValueError("The vLLM config must specify keras_hub_preset to load the correct Keras Hub model.")
            
        self._resolve_vllm_architecture()
        self.model = models.CausalLM.from_preset(self.preset_name, preprocessor=None)

    def _resolve_vllm_architecture(self) -> None:
        """Resolves and sets the corresponding vLLM architecture based on the preset."""
        if hasattr(self.config, "architectures"):
            # Dynamically map the Keras Hub preset architecture to vLLM's architecture if possible
            preset_name = self.preset_name.lower()
            if "gemma" in preset_name:
                self.config.architectures = ["GemmaForCausalLM"]
            elif "llama" in preset_name:
                self.config.architectures = ["LlamaForCausalLM"]
            elif "mistral" in preset_name:
                self.config.architectures = ["MistralForCausalLM"]
            elif "qwen" in preset_name:
                self.config.architectures = ["Qwen2ForCausalLM"]
            else:
                self.config.architectures = ["LlamaForCausalLM"]  # Generic fallback

    def get_vllm_model(self, vllm_config: Any, rng: Any, mesh: Any) -> Tuple:
        """Object-oriented hook for vLLM's TPU runner.
        
        Returns a tuple of functions and configurations expected by vLLM.
        """
        return (
            self._vllm_jit_model,
            self._vllm_compute_logits_fn,
            None,
            None,
            {},
            None,
            self,
        )

    def _vllm_jit_model(self, vllm_config: Any, kv_caches: List[torch.Tensor], input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple:
        """Bound JIT model function for vLLM's execution engine."""
        hidden_states = self.forward_step(
            input_ids, kv_caches, attention_metadata=None
        )
        return kv_caches, hidden_states, None

    def _vllm_compute_logits_fn(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Bound logits computation function for vLLM's execution engine."""
        return self.compute_logits(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embeds input IDs. 
        
        Required by vLLM interface but handled natively within the Keras Hub 
        backbone. We return the input_ids as-is or raise NotImplementedError 
        if explicitly invoked, to prevent silent failures.
        """
        raise NotImplementedError("Embedding is handled internally by Keras Hub backbone.")

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
        is_prompt = getattr(attention_metadata, "is_prompt", False)
        if is_prompt and hasattr(attention_metadata, "seq_lens"):
            seq_lens = attention_metadata.seq_lens
            max_seq_len = max(seq_lens)
            batch_size = len(seq_lens)
            padded_ids = torch.zeros((batch_size, max_seq_len), dtype=input_ids.dtype, device=input_ids.device)
            start_idx = 0
            for i, l in enumerate(seq_lens):
                padded_ids[i, :l] = input_ids[start_idx:start_idx+l]
                start_idx += l
            jax_input_ids = _to_jax(padded_ids)
        else:
            jax_input_ids = _to_jax(input_ids)
            if len(jax_input_ids.shape) == 1:
                jax_input_ids = ops.expand_dims(jax_input_ids, axis=-1)

        x = self.model.backbone.token_embedding(jax_input_ids)
        
        hidden_dim = self.model.backbone.hidden_dim
        x = x * ops.cast(ops.sqrt(hidden_dim), x.dtype)
        
        if is_prompt and hasattr(attention_metadata, "seq_lens"):
            # Use proper padding mask during prefill
            padding_mask_torch = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=input_ids.device)
            for i, l in enumerate(seq_lens):
                padding_mask_torch[i, :l] = True
            padding_mask = _to_jax(padding_mask_torch)
        else:
            padding_mask = ops.ones_like(jax_input_ids, dtype="bool")
        
        # If kv_caches is provided, we use Keras Hub's native caching mechanism.
        # This bridges vLLM's cache manager with Keras Hub's transformer layers.
        # positions usually represents the current cache update index for generation.
        cache_update_index = positions[:, 0] if len(positions.shape) > 1 else positions
        
        updated_kv_caches = []
        for i, layer in enumerate(self.model.backbone.transformer_layers):
            if kv_caches is not None and len(kv_caches) > i:
                # _to_jax will zero-copy wrap the DLPack tensor into a JAX array
                cache_tensor = _to_jax(kv_caches[i])
                x, next_cache = layer(
                    x,
                    padding_mask=padding_mask,
                    cache=cache_tensor,
                    cache_update_index=cache_update_index,
                )
                # Store the updated cache to be returned or updated in-place
                updated_kv_caches.append(next_cache)
            else:
                x = layer(x, padding_mask=padding_mask)
                
        # If we updated caches, we need to map them back to torch tensors
        if updated_kv_caches:
            for i, next_cache in enumerate(updated_kv_caches):
                # We do an in-place copy back to the original PyTorch tensor
                # to satisfy vLLM's memory management
                updated_torch = _to_torch(next_cache, input_ids)
                kv_caches[i].copy_(updated_torch)
            
        hidden_states = self.model.backbone.layer_norm(x)

        if is_prompt and hasattr(attention_metadata, "seq_lens"):
            # Flatten the padded output back to 1D valid tokens only
            hidden_states_torch = _to_torch(hidden_states, input_ids)
            flattened = []
            for i, l in enumerate(seq_lens):
                flattened.append(hidden_states_torch[i, :l, :])
            return torch.cat(flattened, dim=0)

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
