"""The serving model that runs a KerasHub `CausalLM` on vLLM's GPU engine.

`KerasHubTorchModel` is the GPU sibling of `KerasHubVllmModel`: the same
delegation design, implemented against vLLM's torch model protocol instead
of tpu-inference's flax/nnx one. The backbone runs on Keras's torch backend
(`KERAS_BACKEND=torch`), so its tensors are the engine's tensors and no
conversion happens anywhere.

Attention reaches vLLM's paged kernels through the same serving context and
bridge the TPU path uses: the wrapper builds one `vllm.Attention` module per
backbone layer (the engine discovers these and binds paged KV cache to
them), publishes the module list as the context's per-layer caches, and the
bridge hands each attention route "its cache" -- here, its module -- in call
order. The published function simply calls that module.

Importing this module works anywhere (api-gen walks every module); torch
and vLLM are only needed on the serving path.
"""

import math
from typing import NamedTuple

from keras import ops

try:
    from torch import nn

    _TorchModule = nn.Module
except ImportError:  # torch is only present on the GPU serving path
    nn = None
    _TorchModule = object

from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.vllm.context import get_vllm_context
from keras_hub.src.vllm.context import vllm_context_scope


def _attention_cls():
    """Returns vLLM's `Attention` layer class.

    Resolved lazily: it exists only where vllm is installed, and tests
    substitute a recording stand-in here.
    """
    from vllm.model_executor.layers.attention import Attention

    return Attention


# What a block calls its attention layer, by family: Gemma/Gemma 3/Phi-3
# use "attention", Llama/Qwen/Mistral/GPT-NeoX "_self_attention_layer",
# SmolLM3 "self_attn"; "self_attention" is what the encoder-decoder layers
# use, kept for families routed later.
_ATTENTION_ATTRS = (
    "attention",
    "_self_attention_layer",
    "self_attn",
    "self_attention",
)


class _ServedLayer(NamedTuple):
    """One layer's `Attention` module and what it was built with.

    The context carries these as the per-layer caches, so the published
    function can check a route's per-call values against its own layer
    without storing anything on vLLM's module.
    """

    module: object
    scale: float
    window: int | None
    soft_cap: float | None


def _layer_attention_options(block, head_dim):
    """Returns `(scale, sliding_window, soft_cap)` for one transformer block.

    Read off the built attention layer, which is where a family records
    what it does: its softmax scale, whether it windows and how wide, and
    whether it caps its logits. A family that does none of this simply has
    none of the attributes, and one whose attention this cannot find is
    configured from `head_dim` alone rather than wrongly -- the published
    function then rejects any route whose values disagree.

    Args:
        block: A transformer block from the backbone.
        head_dim: The head dimension, used for the softmax scale when the
            layer does not carry one.

    Returns:
        A `(scale, sliding_window, soft_cap)` tuple, where the window and
        the cap are `None` when the layer does not use them.
    """
    attn = next(
        (
            layer
            for name in _ATTENTION_ATTRS
            if (layer := getattr(block, name, None)) is not None
        ),
        None,
    )
    default_scale = head_dim**-0.5
    if attn is None:
        return default_scale, None, None
    # Most families keep their scale here. Gemma computes it in its route
    # instead (it depends on `query_head_dim_normalize`), so it falls back
    # to the usual formula and the published function catches the case
    # where Gemma's differs.
    scale = getattr(attn, "_inv_norm_factor", None) or default_scale
    # Gemma and Qwen alternate windowed and full layers, so a boolean gates
    # the width. Mistral windows every layer and only holds the width.
    window = None
    if getattr(attn, "use_sliding_window_attention", False):
        window = getattr(attn, "sliding_window_size", None)
    elif getattr(attn, "_sliding_window", None):
        window = attn._sliding_window
    return scale, window, getattr(attn, "logit_soft_cap", None)


class KerasHubTorchModel(_TorchModule):
    """Serves a KerasHub `CausalLM` on vLLM's GPU (torch) engine.

    Implements the model protocol vLLM's V1 engine drives -- `__init__`,
    `embed_input_ids`, `forward`, `compute_logits` -- plus `load_preset`,
    which the `keras_hub` load format calls instead of `load_weights`
    (the preset carries the weights; there is no safetensors iterator).

    - `__init__` builds the backbone with a single `CausalLM.from_preset`
      call, then one `vllm.Attention` module per layer, configured from
      what that layer actually holds. The engine finds those modules and
      allocates the paged KV cache against them.
    - `forward` publishes the serving context with the `Attention` modules
      riding as the per-layer caches, then delegates to the backbone; each
      attention layer's route dispatches through the shared bridge to this
      wrapper's published function, which calls the layer's module.
    - `compute_logits` projects through the tied token embedding.

    Sliding windows and attention soft caps are per-layer constructor
    arguments on `vllm.Attention`, while the routes pass them per call.
    Both come from the built layer itself, so the two sides cannot drift:
    the module is constructed from what the layer holds, and the published
    function checks the route's per-call values against it.
    """

    def __init__(self, vllm_config, prefix=""):
        """Builds the backbone and one `Attention` module per layer.

        Args:
            vllm_config: The vLLM config; `model_config.hf_config` carries
                the `keras_hub_preset` written by `setup_vllm_model`.
            prefix: Name prefix for this model's modules. vLLM keys each
                layer's paged KV cache by its `Attention` module's prefix.
        """
        super().__init__()
        self.vllm_config = vllm_config
        hf_config = vllm_config.model_config.hf_config
        self.preset_name = hf_config.keras_hub_preset

        # Serving dtype: vLLM's resolved dtype, same reasoning as the TPU
        # wrapper (vLLM may override the requested dtype, and the KV cache
        # follows the resolved one).
        resolved_dtype = getattr(vllm_config.model_config, "dtype", None)
        if resolved_dtype is None:
            resolved_dtype = getattr(hf_config, "torch_dtype", "bfloat16")
        self._dtype = str(resolved_dtype).removeprefix("torch.")

        num_heads = hf_config.num_attention_heads
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // num_heads

        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # Build the backbone up front, unlike the TPU wrapper: there the
        # loader constructs models under `nnx.eval_shape`, where KerasHub's
        # concrete constructor work cannot run. vLLM's GPU path has no such
        # tracing pass, and building here means each `Attention` module can
        # be configured from the layer it stands for rather than from a
        # description of it.
        model = CausalLM.from_preset(self.preset_name, dtype=self._dtype)
        self.backbone = model.backbone

        # One Attention module per backbone layer. The engine scans the
        # model for these and binds a paged KV cache to each, keyed by
        # `prefix`, so the prefixes must be stable and unique.
        attention = _attention_cls()
        modules, served = [], []
        for i, block in enumerate(self.backbone.transformer_layers):
            scale, window, soft_cap = _layer_attention_options(block, head_dim)
            module = attention(
                num_heads,
                head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                per_layer_sliding_window=window,
                logits_soft_cap=soft_cap,
                cache_config=vllm_config.cache_config,
                prefix=f"{prefix}.layers.{i}.attn",
            )
            modules.append(module)
            served.append(_ServedLayer(module, scale, window, soft_cap))
        # The engine binds KV cache by scanning for Attention modules, so
        # they have to hang off the model. The context carries the records
        # instead, which keeps what each layer was built with on an object
        # this file owns.
        self.layers = nn.ModuleList(modules)
        self._served_layers = served

    def embed_input_ids(self, input_ids):
        """Returns the token embeddings for `input_ids`."""
        return self.backbone.token_embedding(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        """Runs one forward step against vLLM's paged KV cache.

        Publishes the serving context -- the published attention function,
        the per-layer `Attention` modules as the caches, and the engine's
        per-token positions -- then delegates to the backbone. The routes
        and `PositionEmbedding` read the context exactly as they do on TPU.

        Args:
            input_ids: The packed token ids for this step.
            positions: Each token's absolute position in its own request.
            intermediate_tensors: Pipeline-parallel input; unsupported.
            inputs_embeds: Pre-computed embeddings; unsupported.

        Returns:
            The hidden states for the step, one row per token.

        Raises:
            NotImplementedError: If `intermediate_tensors` or
                `inputs_embeds` is given.
            RuntimeError: If the layers did not each dispatch once to the
                paged-attention kernel.
        """
        # Both belong to features this wrapper does not implement:
        # `inputs_embeds` enters below the token embedding, and
        # `intermediate_tensors` is pipeline parallelism. Ignoring either
        # would serve something quietly wrong -- and with `inputs_embeds`,
        # `input_ids` is None, so the failure would be obscure.
        if inputs_embeds is not None or intermediate_tensors is not None:
            raise NotImplementedError(
                "The KerasHub GPU path takes token ids and runs one "
                "pipeline stage; it does not support inputs_embeds or "
                "pipeline parallelism."
            )

        token_ids = input_ids
        if len(token_ids.shape) == 1:
            token_ids = ops.expand_dims(token_ids, axis=-1)
        # vLLM presents already-packed tokens; there is no padding to mask.
        padding_mask = ops.ones_like(token_ids)

        with vllm_context_scope(
            paged_attention_func=self._paged_attention,
            positions=positions,
            kv_caches=self._served_layers,
        ):
            hidden_states = self.backbone(
                {"token_ids": token_ids, "padding_mask": padding_mask},
                training=False,
            )
            # The backbone works in (batch, seq, hidden); the engine wants
            # one row per token, however those two axes were arranged.
            if len(hidden_states.shape) == 3:
                hidden_states = ops.reshape(
                    hidden_states, (-1, hidden_states.shape[-1])
                )

            # Every attention layer must have dispatched exactly once; a
            # mismatch means one silently ran its dense path.
            ctx = get_vllm_context()
            num_layers = len(self.layers)
            if ctx.layer_index != num_layers:
                raise RuntimeError(
                    f"Paged-attention dispatch ran {ctx.layer_index} "
                    f"time(s) for {num_layers} transformer layers. An "
                    "attention layer skipped the vLLM dispatch (or "
                    "dispatched more than once); serving this model would "
                    "produce incorrect output."
                )

        return hidden_states

    def compute_logits(self, hidden_states):
        """Projects hidden states to logits through the tied embedding."""
        return self.backbone.token_embedding(hidden_states, reverse=True)

    def _paged_attention(
        self,
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
        """The published attention function for the GPU path.

        `kv_cache` is this layer's `_ServedLayer`, handed over by the
        bridge in layer-call order. Its module owns the paged KV cache and
        reads vLLM's attention metadata itself, so the call is just
        `module(q, k, v)` on the flat tensors the bridge already built.

        The per-call arguments are checked against what the layer was
        built with; a mismatch means the route and its layer disagree
        about how attention runs.

        Returns:
            A `(kv_cache, attention_output)` tuple. The cache comes back
            as it went in, since the module owns its paged storage.

        Raises:
            RuntimeError: If any of the route's values disagrees with what
                this layer's module was built with.
        """
        expected_scale = kv_cache.scale
        # `isclose` because the two sides reach the same number by
        # different routes (`1 / sqrt(d)` against `d ** -0.5`); the
        # tolerance is far tighter than any real difference in convention,
        # which stays an error.
        if (
            num_heads != self._num_heads
            or num_kv_heads != self._num_kv_heads
            or head_size != self._head_dim
            or not math.isclose(scale, expected_scale, rel_tol=1e-9)
        ):
            raise RuntimeError(
                "The attention route's per-call configuration (heads="
                f"{num_heads}, kv_heads={num_kv_heads}, head_dim="
                f"{head_size}, scale={scale}) does not match what this "
                f"layer's Attention module was built with (heads="
                f"{self._num_heads}, kv_heads={self._num_kv_heads}, "
                f"head_dim={self._head_dim}, scale={expected_scale})."
            )
        expected_window = kv_cache.window
        expected_cap = kv_cache.soft_cap
        if sliding_window != expected_window or soft_cap != expected_cap:
            raise RuntimeError(
                "The attention route passed sliding_window="
                f"{sliding_window}, soft_cap={soft_cap}, but this layer's "
                f"Attention module was built with window={expected_window},"
                f" soft_cap={expected_cap}. The route and the layer it "
                "belongs to disagree."
            )
        return kv_cache, kv_cache.module(q, k, v)
