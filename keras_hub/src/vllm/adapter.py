"""
Adapter module for bridging KerasHub models to vLLM's execution engine.

This module provides the necessary wrappers to translate vLLM's internal state
and PyTorch/XLA tensors into Keras backend-agnostic operations natively via JAX.
"""

import inspect
from typing import Any, Dict, List, Optional, Tuple

import jax
import torch
from keras import ops

from keras_hub import models
from keras_hub.src.vllm.context import clear_vllm_context
from keras_hub.src.vllm.context import set_vllm_context


def _to_jax(tensor: Any) -> Any:
    """Converts a PyTorch tensor to a JAX array via DLPack.

    Args:
        tensor: The input tensor (PyTorch or other framework).

    Returns:
        The JAX array corresponding to the input tensor.

    Raises:
        ValueError: If unwrapping fails.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor

    try:
        return jax.dlpack.from_dlpack(tensor.contiguous())
    except Exception as e:
        if "Unknown device type" in str(e):
            for attr in ["jax_array", "_jax_array", "jax", "unwrap", "data"]:
                if hasattr(tensor, attr):
                    val = getattr(tensor, attr)
                    if callable(val):
                        try:
                            val = val()
                        except Exception:
                            continue
                    if val is not None and "torch.Tensor" not in str(type(val)):
                        return val
            if hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
                import jax.numpy as jnp

                return jnp.asarray(tensor.cpu().detach().numpy())
            raise ValueError(
                f"Failed to unwrap torchax tensor. dir: {dir(tensor)}. type: {type(tensor)}"
            ) from e
        raise


def _to_torch(array: Any, reference_tensor: Optional[torch.Tensor] = None) -> Any:
    """Converts a JAX array back to a PyTorch tensor via DLPack.

    Args:
        array: The JAX array to convert.
        reference_tensor: An optional PyTorch tensor used to infer the target device.

    Returns:
        A PyTorch tensor sharing the same memory as the JAX array.
    """
    if isinstance(array, torch.Tensor):
        return array

    if hasattr(array, "aval"):
        try:
            import torchax

            try:
                import tpu_inference

                if hasattr(tpu_inference, "utils") and hasattr(
                    tpu_inference.utils, "to_torch"
                ):
                    return tpu_inference.utils.to_torch(array)
            except ImportError:
                pass

            if hasattr(torchax, "interop") and hasattr(torchax.interop, "from_jax"):
                return torchax.interop.from_jax(array)

            env = torchax.default_env()
            try:
                return torchax.tensor.Tensor(array, env)
            except Exception:
                return torchax.tensor.Tensor(env, array)
        except Exception:
            return array

    tensor = torch.from_dlpack(array)
    if reference_tensor is not None:
        tensor = tensor.to(reference_tensor.device)
    return tensor


class KerasVLLMAdapter(torch.nn.Module):
    """Adapter that wraps a Keras Hub CausalLM for vLLM.

    This class satisfies vLLM's internal model interface and is instantiated
    when the model prefix is `keras_hub:`. It leverages JAX for execution
    and maps vLLM's paged attention logic into the Keras ecosystem.
    """

    def __init__(
        self,
        vllm_config: Any = None,
        rng: Any = None,
        mesh: Any = None,
        prefix: str = "",
        **kwargs: Any,
    ):
        """Initializes the adapter and loads the underlying KerasHub model.

        ``prefix`` is required for vLLM to recognize this as a "new-style"
        model class (it checks for both ``vllm_config`` and ``prefix`` in the
        signature); otherwise vLLM falls back to old-style arg-guessing and
        fails to pass ``vllm_config``.

        Args:
            vllm_config: Configuration from vLLM including the huggingface config.
            rng: Random number generator context (JAX).
            mesh: Device mesh context for sharding (JAX).
            **kwargs: Additional parameters.

        Raises:
            ValueError: If the `keras_hub_preset` is not specified.
        """
        super().__init__()
        # Extract the underlying huggingface config which we mapped in config.py
        if hasattr(vllm_config, "model_config") and hasattr(
            vllm_config.model_config, "hf_config"
        ):
            self.config = vllm_config.model_config.hf_config
        else:
            self.config = vllm_config

        self.preset_name = getattr(self.config, "keras_hub_preset", None)
        if self.preset_name is None:
            raise ValueError(
                "The vLLM config must specify keras_hub_preset to load the correct model."
            )

        self._resolve_vllm_architecture()

        dtype = getattr(self.config, "torch_dtype", None)
        if dtype == "bfloat16":
            import keras

            keras.config.set_floatx("bfloat16")

        self.model = models.CausalLM.from_preset(
            self.preset_name, preprocessor=None, dtype=dtype
        )

        self.keras_variable_mapping = []
        for i, v in enumerate(self.model.variables):
            name = f"keras_var_{i}"
            self.register_buffer(name, _to_torch(v.value))
            self.keras_variable_mapping.append((v, name))

    def _resolve_vllm_architecture(self) -> None:
        """Resolves and sets the corresponding vLLM architecture based on the preset."""
        if hasattr(self.config, "architectures"):
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
                self.config.architectures = ["LlamaForCausalLM"]

    def get_vllm_model(self, vllm_config: Any, rng: Any, mesh: Any) -> Tuple:
        """Object-oriented hook for vLLM's TPU runner.

        Args:
            vllm_config: The vLLM configuration object.
            rng: Random number generator key.
            mesh: Device mesh.

        Returns:
            A tuple of functions and configurations expected by vLLM.
        """
        model_params = {"keras_variables": [v.value for v in self.model.variables]}
        return (
            self._vllm_jit_model,
            self._vllm_compute_logits_fn,
            model_params,
            None,
            {},
            None,
            self,
        )

    def _vllm_jit_model(
        self,
        params: Dict[str, Any],
        vllm_config: Any,
        kv_caches: List[torch.Tensor],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Any]:
        """Wrapper to evaluate the model step under JIT mapping.

        Returns:
            A tuple containing the updated KV caches, hidden states, and auxiliary output.
        """
        import keras

        state_mapping = zip(self.model.variables, params["keras_variables"])
        with keras.StatelessScope(state_mapping=state_mapping):
            hidden_states, updated_kv_caches = self.forward_step(
                input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attention_metadata=attention_metadata,
            )
        return updated_kv_caches, hidden_states, None

    def _vllm_compute_logits_fn(
        self, params: Dict[str, Any], hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Bound logits computation function for vLLM's execution engine."""
        import keras

        state_mapping = zip(self.model.variables, params["keras_variables"])
        with keras.StatelessScope(state_mapping=state_mapping):
            return self.compute_logits(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embeds input IDs.

        Required by vLLM interface but handled natively within the Keras Hub
        backbone. We return the input_ids as-is or raise NotImplementedError
        if explicitly invoked, to prevent silent failures.
        """
        raise NotImplementedError("Embedding is handled internally by Keras Hub backbone.")

    def _get_state_mapping(self) -> List[Tuple[Any, Any]]:
        """Builds the Keras StatelessScope state mapping from the patched buffers.

        The weight buffers are immutable after load, so the (variable -> JAX
        array) mapping is built once and cached. Without this, every decode step
        re-ran a DLPack conversion over all of the model's weight buffers
        (hundreds of tensors for a 2B model) on the host before any TPU compute
        could start — a dominant per-token overhead.
        """
        cached = getattr(self, "_state_mapping_cache", None)
        if cached is None:
            cached = [
                (v, _to_jax(getattr(self, name)))
                for v, name in self.keras_variable_mapping
            ]
            self._state_mapping_cache = cached
        return cached

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass for vLLM.

        Supports both GPU and TPU signature structures dynamically.
        """
        is_tpu = False
        try:
            from tpu_inference.models.vllm.vllm_model_wrapper_context import (
                get_vllm_model_wrapper_context,
            )

            vllm_ctx = get_vllm_model_wrapper_context()
            is_tpu = vllm_ctx is not None
        except (ImportError, AssertionError):
            pass

        # VllmModelWrapper invokes the model as `model(**kwargs)`, so inputs
        # arrive as keyword args; fall back to positional for other callers.
        def _arg(idx, name):
            if len(args) > idx:
                return args[idx]
            return kwargs.get(name)

        if is_tpu:
            input_ids = _arg(0, "input_ids")
            positions = _arg(1, "positions")
            try:
                from vllm.forward_context import (
                    get_forward_context,
                    is_forward_context_available,
                )

                if is_forward_context_available():
                    fc = get_forward_context()
                    attn_metadata = fc.attn_metadata
                else:
                    attn_metadata = _arg(3, "attn_metadata")
            except ImportError:
                attn_metadata = _arg(3, "attn_metadata")
            kv_caches = vllm_ctx.kv_caches if vllm_ctx is not None else None
        else:
            input_ids = _arg(0, "input_ids")
            positions = _arg(1, "positions")
            kv_caches = _arg(2, "kv_caches")
            attn_metadata = _arg(3, "attn_metadata")

        import keras

        with keras.StatelessScope(state_mapping=self._get_state_mapping()):
            hidden_states, updated_kv_caches = self.forward_step(
                input_ids, positions, kv_caches, attn_metadata
            )

        # On the TPU path vLLM harvests the new KV caches from the model-wrapper
        # context, so we must write the kernel-updated caches back into it.
        # Otherwise every decode step reuses the original (stale) cache.
        if is_tpu and updated_kv_caches is not None and vllm_ctx is not None:
            wrapper_caches = getattr(vllm_ctx, "kv_caches", None)
            if wrapper_caches is not None:
                for i, updated in enumerate(updated_kv_caches):
                    if updated is not None and i < len(wrapper_caches):
                        wrapper_caches[i] = updated
        return hidden_states

    def _get_wrapper_mesh(self) -> Any:
        """Returns the JAX device mesh from vLLM's TPU model-wrapper context.

        The Pallas paged-attention kernel shards across this mesh. Returns
        ``None`` when not running under the tpu-inference wrapper (e.g. on the
        GPU path or in unit tests), in which case the kernel is not invoked.
        """
        try:
            from tpu_inference.models.vllm.vllm_model_wrapper_context import (
                get_vllm_model_wrapper_context,
            )

            wrapper_ctx = get_vllm_model_wrapper_context()
            return getattr(wrapper_ctx, "mesh", None) if wrapper_ctx else None
        except (ImportError, AssertionError):
            return None

    @staticmethod
    def _resolve_backbone_component(backbone: Any, names: List[str], what: str) -> Any:
        """Returns the first attribute in ``names`` present on ``backbone``.

        Raises a clear error if none are found, so an unsupported backbone fails
        loudly rather than with a cryptic AttributeError mid-forward.
        """
        for name in names:
            component = getattr(backbone, name, None)
            if component is not None:
                return component
        raise AttributeError(
            f"KerasVLLMAdapter could not find the {what} on backbone "
            f"{type(backbone).__name__} (tried {names}). The vLLM forward "
            f"re-implements the backbone and expects a standard KerasHub "
            f"decoder layout (token_embedding, transformer_layers, final norm)."
        )

    def forward_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attention_metadata: Any,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Triggers a single autoregressive step."""
        is_prompt = getattr(attention_metadata, "is_prompt", False)

        jax_input_ids = _to_jax(input_ids)
        jax_positions = _to_jax(positions)

        if len(jax_input_ids.shape) == 1:
            jax_input_ids = ops.expand_dims(jax_input_ids, axis=-1)

        # This method re-implements the backbone forward so per-layer paged KV
        # caches can be threaded into each attention layer. It assumes a standard
        # KerasHub decoder layout: a `token_embedding`, an iterable of
        # `transformer_layers`, and a final norm; optionally a learned
        # `position_embedding`. Components are resolved defensively so an
        # unsupported backbone fails with a clear message instead of an
        # AttributeError deep in the loop.
        backbone = self.model.backbone
        token_embedding = self._resolve_backbone_component(
            backbone, ["token_embedding"], "token embedding"
        )
        x = token_embedding(jax_input_ids)

        hidden_dim = backbone.hidden_dim
        is_gemma = (
            getattr(self, "preset_name", "") and "gemma" in self.preset_name.lower()
        )
        if is_gemma:
            x = x * ops.sqrt(ops.cast(hidden_dim, x.dtype))

        # Add learned position embeddings for backbones that use them (GPT-2,
        # OPT, ...). RoPE-based backbones (Gemma, Llama, Mistral) have no
        # `position_embedding` layer and apply rotary positions inside attention,
        # so this is skipped for them. We index by vLLM's `positions` tensor
        # (not 0..T-1) so decode steps use the correct absolute position.
        position_embedding = getattr(backbone, "position_embedding", None)
        if position_embedding is not None and jax_positions is not None:
            pos_weight = position_embedding.position_embeddings
            pos_ids = ops.cast(ops.reshape(jax_positions, (-1,)), "int32")
            pos_emb = ops.take(pos_weight, pos_ids, axis=0)
            pos_emb = ops.reshape(pos_emb, ops.shape(x))
            x = x + ops.cast(pos_emb, x.dtype)

        block_tables = None
        if (
            hasattr(attention_metadata, "block_tables")
            and attention_metadata.block_tables is not None
        ):
            block_tables = _to_jax(attention_metadata.block_tables)

        slot_mapping = None
        if hasattr(attention_metadata, "slot_mapping_tensor"):
            slot_mapping = _to_jax(attention_metadata.slot_mapping_tensor)
        elif (
            hasattr(attention_metadata, "slot_mapping")
            and attention_metadata.slot_mapping is not None
        ):
            slot_mapping = _to_jax(attention_metadata.slot_mapping)

        # Retrieve the vLLM native paged attention function if injected from TPU inference.
        # This prevents Keras from maintaining complex hardware-specific kernels.
        paged_attn_func = getattr(attention_metadata, "paged_attention_func", None)
        # The Pallas kernel needs the JAX device mesh, which lives on the vLLM
        # model-wrapper context on the TPU path.
        mesh = self._get_wrapper_mesh()
        set_vllm_context(
            block_tables,
            slot_mapping,
            attention_metadata,
            paged_attn_func,
            mesh,
            positions=jax_positions,
        )

        try:
            kwargs_base = {}
            if jax_positions is not None:
                kwargs_base["positions"] = jax_positions

            if hasattr(attention_metadata, "seq_lens_tensor"):
                kwargs_base["seq_lens"] = _to_jax(attention_metadata.seq_lens_tensor)
            elif (
                hasattr(attention_metadata, "seq_lens")
                and attention_metadata.seq_lens is not None
            ):
                kwargs_base["seq_lens"] = ops.convert_to_tensor(
                    attention_metadata.seq_lens, dtype="int32"
                )

            if is_prompt and hasattr(attention_metadata, "seq_lens"):
                padding_mask = ops.ones(ops.shape(jax_input_ids), dtype="bool")
            else:
                padding_mask = ops.ones(ops.shape(jax_input_ids), dtype="bool")

            kwargs_base["padding_mask"] = padding_mask

            jax_kv_caches = None
            if kv_caches is not None:
                jax_kv_caches = [_to_jax(cache) for cache in kv_caches]

            transformer_layers = self._resolve_backbone_component(
                backbone, ["transformer_layers"], "transformer layers"
            )
            # All decoder layers share a class/signature, so inspect once and
            # cache on the instance — the layout is fixed for the model's
            # lifetime, so this need not run per layer or per forward step.
            layer_params = getattr(self, "_layer_params_cache", None)
            if layer_params is None:
                layer_params = (
                    set(inspect.signature(transformer_layers[0].call).parameters)
                    if len(transformer_layers)
                    else set()
                )
                self._layer_params_cache = layer_params
            for i, layer in enumerate(transformer_layers):
                kwargs = kwargs_base.copy()

                cache_tensor = (
                    jax_kv_caches[i]
                    if jax_kv_caches is not None and len(jax_kv_caches) > i
                    else None
                )

                if "self_attention_cache" in layer_params:
                    kwargs["self_attention_cache"] = cache_tensor
                elif "cache" in layer_params:
                    if hasattr(layer, "decoder_sequence_length"):
                        kwargs["cache_update_index"] = 0
                    else:
                        kwargs["cache"] = cache_tensor

                if "kv_cache" in layer_params:
                    kwargs["kv_cache"] = cache_tensor

                supported_kwargs = {
                    k: v for k, v in kwargs.items() if k in layer_params
                }

                out = layer(x, **supported_kwargs)

                if isinstance(out, tuple):
                    x = out[0]
                    if len(out) > 1 and jax_kv_caches is not None:
                        jax_kv_caches[i] = out[1]
                else:
                    x = out

            final_norm = self._resolve_backbone_component(
                backbone,
                ["layer_norm", "final_layer_norm", "norm"],
                "final norm",
            )
            hidden_states = final_norm(x)

            if len(hidden_states.shape) == 3 and hidden_states.shape[1] == 1:
                hidden_states = ops.squeeze(hidden_states, axis=1)

            return _to_torch(hidden_states, input_ids), jax_kv_caches
        finally:
            clear_vllm_context()

    def load_weights(self, weights: Any = None) -> None:
        """Loads weights. Pre-loaded by Keras Hub natively."""
        pass

    def compute_logits(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Computes logits for the LM head."""
        jax_hidden_states = _to_jax(hidden_states)
        logits = self.model.backbone.token_embedding(
            jax_hidden_states, reverse=True
        )
        return _to_torch(logits, hidden_states)
