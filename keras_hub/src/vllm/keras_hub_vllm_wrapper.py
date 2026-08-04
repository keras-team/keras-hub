"""The serving model that runs a KerasHub `CausalLM` on vLLM's TPU backend.

`KerasHubVllmModel` lives here in keras-hub and is registered with
tpu-inference's model registry (and vLLM's) by tpu-inference's plugin hook,
so the native `flax_nnx` loader resolves it by architecture name like any
other model. It is only ever instantiated on the serving path, where flax
and tpu-inference are installed; importing it works anywhere (api-gen walks
every module), with the nnx base class swapped in only when flax exists.
"""

from keras import ops

try:
    from flax import nnx

    _NnxModule = nnx.Module
except ImportError:  # flax is only present on the serving path
    _NnxModule = object

from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.vllm.context import get_vllm_context
from keras_hub.src.vllm.context import vllm_context_scope


class KerasHubVllmModel(_NnxModule):
    """Serves a KerasHub `CausalLM` on tpu-inference's native JAX path.

    Registered under the ``"KerasHubForCausalLM"`` architecture string
    (``KERAS_HUB_ARCHITECTURE``), which follows the HF naming convention
    configs use on the wire; the class itself carries the keras-hub-style
    name.

    An adapter, not a conversion: it implements the model interface the
    native `flax_nnx` runner drives, reusing the
    preset's existing backbone and weights. Keras's NNX mode
    (`KERAS_NNX_ENABLED=true`) makes the backbone's variables nnx state, so
    the runner's `nnx.split`/`nnx.merge` machinery carries the weights with
    no conversion:

    - `__init__` records the preset name and dtype; it builds nothing, so
      it is safe under the loader's `nnx.eval_shape` abstract pass.
    - `load_weights` builds the model and loads the preset weights with a
      single `CausalLM.from_preset` call. The loader always calls it
      eagerly, so KerasHub construction and weight IO are unrestricted.
    - `__call__` runs one forward step: it publishes the serving context
      (the paged-attention kernel, the per-layer paged KV caches, and
      vLLM's per-token positions), then delegates to the backbone's own
      forward; each attention layer's vLLM route reads that context and
      dispatches to the paged-attention kernel.
    - `compute_logits` projects hidden states to vocabulary logits through
      the tied token embedding.
    """

    def __init__(self, vllm_config, rng_key, mesh=None):
        """Records what `load_weights` needs; builds nothing.

        The loader constructs models under `nnx.eval_shape`, where concrete
        tensor work cannot run (KerasHub constructors compute values such
        as Gemma's embedding scale). Keeping `__init__` free of any Keras
        construction makes it safe to trace; the model itself is built in
        `load_weights`, which the loader always calls eagerly.

        Args:
            vllm_config: The vLLM config; `model_config.hf_config` carries
                the `keras_hub_preset` written by `setup_vllm_model`.
            rng_key: JAX PRNG key (unused; weights come from the preset).
            mesh: JAX device mesh used by the paged-attention kernel.
        """
        self.vllm_config = vllm_config
        self.mesh = mesh
        self.preset_name = vllm_config.model_config.hf_config.keras_hub_preset
        # Serving dtype is the one vLLM resolved, not the one requested.
        # `model_config.dtype` is authoritative: vLLM may override the
        # `torch_dtype` written into config.json (notably downcasting float32
        # under dtype="auto"), and the KV cache it allocates follows the
        # resolved dtype. Building the backbone from the requested value
        # instead would feed the paged kernel q/k/v in a dtype its cache does
        # not match. Falls back to `torch_dtype` if vLLM exposes no dtype.
        resolved_dtype = getattr(vllm_config.model_config, "dtype", None)
        if resolved_dtype is None:
            resolved_dtype = getattr(
                vllm_config.model_config.hf_config, "torch_dtype", "bfloat16"
            )
        # `str(torch.bfloat16)` is "torch.bfloat16"; Keras wants "bfloat16".
        self._dtype = str(resolved_dtype).removeprefix("torch.")

    def load_weights(self, *args, **kwargs):
        """Builds the KerasHub model and loads the preset weights.

        One `CausalLM.from_preset` call, the same as any other KerasHub
        usage: it builds the model and loads the preset weights in place.
        Only the backbone is kept: serving never calls the task wrapper,
        and a compiled task carries a layer-keyed dict (Keras 3.15's
        `_compiled_trainable_state`) whose keys nnx's graph flatten cannot
        sort.

        Args:
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).
        """
        model = CausalLM.from_preset(self.preset_name, dtype=self._dtype)
        self.backbone = model.backbone

    def __call__(
        self,
        kv_caches,
        input_ids,
        attention_metadata,
        *args,
        **kwargs,
    ):
        """Runs one forward step with vLLM's paged KV cache.

        Publishes the serving context (the paged-attention kernel, the
        per-layer paged KV caches, and vLLM's per-token positions), then
        delegates to the KerasHub backbone's own forward, which keeps every
        model-specific detail (embedding scaling, learned vs. rotary
        positions, per-family norms) in the model. Each attention layer's
        vLLM route and KerasHub's `PositionEmbedding` read the context in
        place; the scope clears it when the step finishes, even on error.

        Returns:
            `(updated_kv_caches, hidden_states, None, None)` — the tuple
            the native runner expects.
        """
        positions = getattr(attention_metadata, "input_positions", None)
        if positions is None:
            positions = kwargs.get("positions")

        token_ids = input_ids
        if len(token_ids.shape) == 1:
            token_ids = ops.expand_dims(token_ids, axis=-1)
        # vLLM presents already-packed tokens; there is no padding to mask.
        padding_mask = ops.ones_like(token_ids)

        with self._serving_context(kv_caches, attention_metadata, positions):
            hidden_states = self.backbone(
                {"token_ids": token_ids, "padding_mask": padding_mask},
                training=False,
            )
            # Tokens ride as (num_tokens, 1); drop the seq axis the runner
            # does not expect.
            if len(hidden_states.shape) == 3 and hidden_states.shape[1] == 1:
                hidden_states = ops.squeeze(hidden_states, axis=1)

            # The bridge stored each layer's kernel-updated cache here. It
            # is None on a cacheless pass (e.g. vLLM's startup profiling
            # run).
            ctx = get_vllm_context()
            updated_kv_caches = (
                list(ctx.updated_kv_caches)
                if ctx.updated_kv_caches is not None
                else None
            )
            # Every attention layer must have dispatched to the kernel
            # exactly once; a mismatch means one silently ran its dense
            # path (or ran twice), which would corrupt the paged cache.
            caches = ctx.kv_caches
            num_caches = len(caches) if caches is not None else None
            if num_caches is not None and ctx.layer_index != num_caches:
                raise RuntimeError(
                    f"Paged-attention dispatch ran {ctx.layer_index} "
                    f"time(s) for {num_caches} transformer layers. An "
                    "attention layer skipped the vLLM dispatch (or "
                    "dispatched more than once); serving this model would "
                    "produce incorrect output."
                )

        return updated_kv_caches, hidden_states, None, None

    def compute_logits(self, hidden_states, *args, **kwargs):
        """Projects hidden states to vocab logits via the tied embedding.

        Args:
            hidden_states: Tensor. A tensor of hidden states from the
                backbone.
            *args: Variable length argument list (unused).
            **kwargs: Arbitrary keyword arguments (unused).

        Returns:
            Tensor. A tensor of logits.
        """
        return self.backbone.token_embedding(hidden_states, reverse=True)

    def _serving_context(self, kv_caches, attention_metadata, positions):
        """Builds the serving-context scope for one forward step.

        Returns keras-hub's ``vllm_context_scope`` context manager, which
        publishes the thread-local serving context on entry and always
        clears it on exit, even when the forward raises.

        The published function wraps tpu-inference's ragged-paged-attention
        kernel (`_jax_attn_func`) behind a small stable contract carrying
        only what a KerasHub attention layer knows::

            fn(kv_cache, q, k, v, scale, head_size, num_heads, num_kv_heads,
               sliding_window=None, soft_cap=None) -> (new_kv_cache, output)

        Engine-side arguments (the attention metadata, the mesh) are closed
        over here, so kernel signature changes stay local to this repo. The
        per-layer paged KV caches ride in the context too: KerasHub's shared
        bridge consumes them in layer-call order, so the layers run their
        plain `cache=None` path and need no cache threading.
        """
        # tpu-inference is imported here, not at module top: it is an
        # optional dependency of keras-hub, and the tpu-inference plugin
        # hook imports this module, so a top-level import would be
        # circular. A missing kernel fails loudly with the real cause.
        try:
            from tpu_inference.layers.vllm.backends.flash_attn import (
                _jax_attn_func,
            )
        except ImportError as e:
            raise ImportError(
                "Serving a KerasHub model requires tpu-inference "
                "(vllm-tpu), which provides the paged-attention kernel."
            ) from e

        mesh = self.mesh

        def paged_attention(
            kv_cache,
            q,
            k,
            v,
            scale,
            head_size,
            num_heads,
            num_kv_heads,
            sliding_window=None,
            soft_cap=None,
        ):
            return _jax_attn_func(
                kv_cache=kv_cache,
                q=q,
                k=k,
                v=v,
                sinks=None,  # KerasHub attention layers have no sinks.
                attention_metadata=attention_metadata,
                shared_attention_metadata=None,  # single-chip serving
                mesh=mesh,
                scale=scale,
                head_size=head_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
            )

        return vllm_context_scope(
            paged_attention_func=paged_attention,
            positions=positions,
            kv_caches=kv_caches,
        )
