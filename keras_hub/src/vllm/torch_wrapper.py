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
bridge hands each attention route "its cache" — here, its module — in call
order. The published function simply calls that module.

Importing this module works anywhere (api-gen walks every module); torch
and vLLM are only needed on the serving path.
"""

from keras import ops

try:
    from torch import nn

    _TorchModule = nn.Module
except ImportError:  # torch is only present on the GPU serving path
    nn = None
    _TorchModule = object

try:
    from vllm.model_executor.model_loader import BaseModelLoader
except ImportError:  # vllm is only present on the serving path
    BaseModelLoader = object

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


class KerasHubTorchModel(_TorchModule):
    """Serves a KerasHub `CausalLM` on vLLM's GPU (torch) engine.

    Implements the model protocol vLLM's V1 engine drives — `__init__`,
    `embed_input_ids`, `forward`, `compute_logits` — plus `load_preset`,
    which the `keras_hub` load format calls instead of `load_weights`
    (the preset carries the weights; there is no safetensors iterator).

    - `__init__` builds one `vllm.Attention` module per backbone layer from
      the dims in the config. The engine finds these modules and allocates
      the paged KV cache against them; the backbone itself is not built
      here.
    - `load_preset` builds the model and loads the preset weights with a
      single `CausalLM.from_preset` call, on the torch backend.
    - `forward` publishes the serving context with the `Attention` modules
      riding as the per-layer caches, then delegates to the backbone; each
      attention layer's route dispatches through the shared bridge to this
      wrapper's published function, which calls the layer's module.
    - `compute_logits` projects through the tied token embedding.

    Sliding windows and attention soft caps are per-layer constructor
    arguments on `vllm.Attention`, not per-call arguments, so families that
    use them (Gemma 2/3, windowed Qwen presets) need config-side support
    that this first GPU version does not carry. Their routes pass the
    per-call values, and the published function fails loudly rather than
    serving with a silently missing window or cap.
    """

    def __init__(self, vllm_config, prefix=""):
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

        num_layers = hf_config.num_hidden_layers
        num_heads = hf_config.num_attention_heads
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // num_heads

        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._scale = head_dim**-0.5

        # One Attention module per backbone layer. The engine scans the
        # model for these and binds a paged KV cache to each, keyed by
        # `prefix`, so the prefixes must be stable and unique.
        attention = _attention_cls()
        self.layers = nn.ModuleList(
            [
                attention(
                    num_heads,
                    head_dim,
                    scale=self._scale,
                    num_kv_heads=num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    prefix=f"{prefix}.layers.{i}.attn",
                )
                for i in range(num_layers)
            ]
        )

    def load_preset(self):
        """Builds the KerasHub model and loads the preset weights.

        Called by the `keras_hub` load format's loader. One
        `CausalLM.from_preset` call, like any other KerasHub usage; only
        the backbone is kept, since serving never calls the task wrapper.
        """
        model = CausalLM.from_preset(self.preset_name, dtype=self._dtype)
        self.backbone = model.backbone

    def embed_input_ids(self, input_ids):
        return self.backbone.token_embedding(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        """Runs one forward step against vLLM's paged KV cache.

        Publishes the serving context — the published attention function,
        the per-layer `Attention` modules as the caches, and the engine's
        per-token positions — then delegates to the backbone. The routes
        and `PositionEmbedding` read the context exactly as they do on TPU.
        """
        token_ids = input_ids
        if len(token_ids.shape) == 1:
            token_ids = ops.expand_dims(token_ids, axis=-1)
        # vLLM presents already-packed tokens; there is no padding to mask.
        padding_mask = ops.ones_like(token_ids)

        with vllm_context_scope(
            paged_attention_func=self._paged_attention,
            positions=positions,
            kv_caches=list(self.layers),
        ):
            hidden_states = self.backbone(
                {"token_ids": token_ids, "padding_mask": padding_mask},
                training=False,
            )
            # Tokens ride as (num_tokens, 1); drop the seq axis the engine
            # does not expect.
            if len(hidden_states.shape) == 3 and hidden_states.shape[1] == 1:
                hidden_states = ops.squeeze(hidden_states, axis=1)

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

        `kv_cache` is this layer's `vllm.Attention` module (the context
        carries the module list as the per-layer caches). The module owns
        the paged KV cache and reads the attention metadata from vLLM's
        forward context, so the call is just `module(q, k, v)` on the flat
        `(num_tokens, heads * head_dim)` tensors the bridge already built.

        The per-call arguments exist to cross-check the module's
        construction-time configuration: a mismatch means the config wrote
        different dims than the backbone actually has, or the family needs
        per-layer options this GPU version does not support yet.
        """
        if (
            num_heads != self._num_heads
            or num_kv_heads != self._num_kv_heads
            or head_size != self._head_dim
            or scale != self._scale
        ):
            raise RuntimeError(
                "The attention route's per-call configuration (heads="
                f"{num_heads}, kv_heads={num_kv_heads}, head_dim="
                f"{head_size}, scale={scale}) does not match the config "
                f"this wrapper built its Attention modules from (heads="
                f"{self._num_heads}, kv_heads={self._num_kv_heads}, "
                f"head_dim={self._head_dim}, scale={self._scale})."
            )
        if sliding_window is not None or soft_cap is not None:
            raise RuntimeError(
                "Sliding-window and soft-cap attention are per-layer "
                "constructor options on vLLM's GPU Attention layer and are "
                "not supported by the GPU path yet; this family serves on "
                "the TPU path."
            )
        return kv_cache, kv_cache(q, k, v)


class KerasHubPresetLoader(BaseModelLoader):
    """The `keras_hub` load format: weights come from the preset.

    The model directory `setup_vllm_model` writes holds only a config, so
    the stock loaders have nothing to stream — and `load_format="dummy"`
    would randomize the backbone's parameters after loading. This loader
    delegates to the model's own `load_preset`, which is one
    `CausalLM.from_preset` call.
    """

    def download_model(self, model_config):
        # from_preset downloads on demand; nothing to prefetch here.
        pass

    def load_weights(self, model, model_config):
        model.load_preset()
