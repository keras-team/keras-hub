"""
Registry module for Keras Hub to vLLM Integration.

Materializes a model directory for `LLM(model="keras_hub:preset_name")` so that
tpu-inference's native flax/nnx path serves it via `KerasHubVllmModel`.
"""

import inspect
import json
import logging
import os
import tempfile

import keras

# vLLM is an optional dependency of KerasHub, and `import keras_hub` imports
# this module (via the generated `keras_hub.vllm` API) — so it must import
# cleanly without vLLM. The serving class is defined only when vLLM (with its
# tokenizer registry) is present; otherwise a stub raises with install
# instructions on use. Same optional-import pattern as the tensorflow and
# kagglehub guards in `tensor_utils.py` / `preset_utils.py`.
try:
    from vllm import LLM as _BaseLLM
    from vllm import SamplingParams
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
except ImportError:  # vllm not installed, or too old to serve KerasHub
    _BaseLLM = None
    SamplingParams = None
    RENDERER_REGISTRY = None
    TokenizerRegistry = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.samplers.random_sampler import RandomSampler
from keras_hub.src.samplers.serialization import get as get_sampler
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.samplers.top_p_sampler import TopPSampler
from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.vllm.hf_config import KERAS_HUB_ARCHITECTURE
from keras_hub.src.vllm.hf_config import KERAS_HUB_MODEL_TYPE
from keras_hub.src.vllm.hf_config import register_hf_config

# We write our own neutral architecture/model_type (no borrowed model family):
# `KerasHubForCausalLM` / `keras_hub` (see `hf_config`). `register_hf_config`
# teaches transformers the `model_type`, and tpu-inference registers the
# architecture with vLLM. Both only carry KV dims for cache profiling; the real
# model is selected by `keras_hub_preset` and served by tpu-inference's
# `KerasHubVllmModel` model class.


def _normalize_dtype(dtype):
    """Normalizes any dtype spelling to a plain string for config.json.

    Accepts the forms users actually pass: None or "auto" (use the
    default), a keras DTypePolicy (use its name, e.g. "mixed_bfloat16"),
    a framework dtype (torch.bfloat16, np.float32), or a plain string.
    """
    if dtype is None or dtype == "auto":
        return "bfloat16"
    if hasattr(dtype, "name"):  # keras DTypePolicy or tf dtype
        dtype = dtype.name
    if not isinstance(dtype, str):
        dtype = keras.backend.standardize_dtype(dtype)
    if dtype.startswith("mixed_"):
        dtype = dtype.removeprefix("mixed_")
    return dtype


def setup_vllm_model(preset, dtype="bfloat16", max_model_len=None):
    """Creates a configuration directory for vLLM to load a Keras Hub preset.

    Writes a ``config.json`` with the ``KerasHubForCausalLM`` architecture
    (registered with vLLM and transformers by tpu-inference, so validation
    and loading resolve like any other model), the preset's real dims (vLLM
    sizes the KV cache from them), and a ``keras_hub_preset`` field naming
    the preset for tpu-inference's model class to build. Tokenization is
    served by the preset's own tokenizer (``tokenizer_mode="keras_hub"``).

    Args:
        preset: The Keras Hub preset name (e.g., "gemma_2b_en").
        dtype: The torch dtype to run inference with.
        max_model_len: Optional context-length ceiling to write as
            `max_position_embeddings`. RoPE presets don't serialize their
            limit, so they default to 8192; pass this to serve longer
            contexts.

    Returns:
        A ``tempfile.TemporaryDirectory`` whose ``name`` is the directory to
        pass to ``vllm.LLM``. The directory is deleted when the returned
        object is garbage collected, so hold a reference to it for as long
        as the engine needs the files.
    """
    dtype = _normalize_dtype(dtype)
    temp_dir = tempfile.TemporaryDirectory(prefix="keras_hub_vllm_")

    # Teach transformers our neutral `keras_hub` model_type in this process, so
    # `AutoConfig.from_pretrained` can read the config we write below. (Serving
    # workers register it again via tpu-inference's import.)
    register_hf_config()

    # Serving tokenizes with this same tokenizer (`tokenizer_mode=
    # "keras_hub"`), so it must load — fail here at setup, with the real
    # error, rather than somewhere inside vLLM's engine. It also supplies
    # vocab_size (vLLM profiles KV-cache memory from it) and eos.
    tokenizer = Tokenizer.from_preset(preset)
    vocab_size = tokenizer.vocabulary_size()
    if vocab_size is None:
        raise ValueError(
            f"The tokenizer for preset '{preset}' returned None for "
            "vocabulary_size(). vLLM needs the real vocabulary size to "
            "profile KV-cache memory."
        )
    vocab_size = int(vocab_size)
    eos_token_id = getattr(tokenizer, "end_token_id", None)

    # Read the backbone's real dims once (parses the preset config.json),
    # used to size the KV cache below.
    arch_config = _derive_arch_config(preset)

    config_dict = {
        "architectures": [KERAS_HUB_ARCHITECTURE],
        "_name_or_path": f"keras_hub:{preset}",
        "keras_hub_preset": preset,
        "torch_dtype": dtype,
        "model_type": KERAS_HUB_MODEL_TYPE,
        "vocab_size": vocab_size,
    }
    if eos_token_id is not None:
        config_dict["eos_token_id"] = int(eos_token_id)
    # Only write bos when the preset really defines a start token, so the
    # config always agrees with what `KerasHubTokenizer.bos_token_id`
    # reports (e.g. Qwen has eos but no bos). Explicit None check, not
    # `or`: a start_token_id of 0 is valid.
    start_token_id = getattr(tokenizer, "start_token_id", None)
    if start_token_id is not None:
        config_dict["bos_token_id"] = int(start_token_id)

    # Write the REAL architecture dims so vLLM allocates a matching KV cache.
    # Without these, vLLM guesses from `model_type` (e.g. "gemma2" -> HF
    # Gemma-2 defaults), which mismatches the actual KerasHub backbone and fails
    # the kernel's kv_cache-shape check (num_kv_heads / num_layers / head_dim).
    # This intentionally overrides the tokenizer-derived vocab_size above with
    # the backbone's embedding vocab (they differ when a model pads its
    # embedding table), which is what vLLM's logits layer must match.
    config_dict.update(arch_config)
    if max_model_len is not None:
        config_dict["max_position_embeddings"] = int(max_model_len)

    with open(
        os.path.join(temp_dir.name, "config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config_dict, f)

    return temp_dir


def _derive_arch_config(preset):
    """Reads the KerasHub backbone config and maps it to HF/vLLM config keys.

    Reads the preset's serialized ``config.json`` directly (no model build, no
    TPU allocation) and translates its backbone args to the fields vLLM uses
    for KV-cache allocation. Handles the naming conventions generically:
    multi-head configs expose ``num_heads``; grouped/multi-query configs
    expose ``num_query_heads`` / ``num_key_value_heads`` / ``head_dim``; and
    some (SmolLM3) use HF's ``num_attention_heads``.

    A missing config would make vLLM guess dims from ``model_type`` and
    mis-size the KV cache, so a config that fails to load raises here.
    """
    cfg = load_json(preset).get("config", {})

    hidden = cfg.get("hidden_dim")
    n_heads = (
        cfg.get("num_query_heads")
        or cfg.get("num_heads")
        or cfg.get("num_attention_heads")
    )
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim")
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads

    arch = {}
    if cfg.get("num_layers"):
        arch["num_hidden_layers"] = int(cfg["num_layers"])
    if n_heads:
        arch["num_attention_heads"] = int(n_heads)
    if n_kv:
        arch["num_key_value_heads"] = int(n_kv)
    if head_dim:
        arch["head_dim"] = int(head_dim)
    if hidden:
        arch["hidden_size"] = int(hidden)
    if cfg.get("intermediate_dim"):
        arch["intermediate_size"] = int(cfg["intermediate_dim"])
    if cfg.get("vocabulary_size"):
        arch["vocab_size"] = int(cfg["vocabulary_size"])
    # vLLM caps max_model_len at max_position_embeddings. Learned-position
    # presets (GPT-2) serialize max_sequence_length; write it so a sequence
    # can't index past the position table. RoPE presets don't serialize it
    # and fall to KerasHubConfig's 8192 default — pass `max_model_len` to
    # `KerasHubLLM` to raise that ceiling.
    if cfg.get("max_sequence_length"):
        arch["max_position_embeddings"] = int(cfg["max_sequence_length"])
    # Fail here with the real cause: without these, vLLM would die later
    # inside KV-cache profiling with a far less useful error.
    required = ["num_hidden_layers", "num_attention_heads", "hidden_size"]
    missing = [key for key in required if key not in arch]
    if missing:
        raise ValueError(
            f"The config for preset '{preset}' is missing fields needed "
            f"for vLLM serving: {missing}. Is this a CausalLM backbone?"
        )
    return arch


def sampler_to_sampling_kwargs(sampler):
    """Maps a KerasHub sampler to equivalent `vllm.SamplingParams` kwargs.

    KerasHub models carry their own sampler (set via `compile(sampler=...)`,
    default `"top_k"`). vLLM samples in the engine instead, so serving honors
    the model's sampler by translating it into the default `SamplingParams`.

    Args:
        sampler: A `keras_hub.samplers.Sampler` instance.

    Returns:
        A dict of `SamplingParams` kwargs, or `None` when the sampler has no
        vLLM equivalent (beam, contrastive, speculative) — vLLM's own
        defaults apply in that case.
    """
    if sampler is None:
        return None
    # Matched by isinstance, so user subclasses map by behavior regardless
    # of their class names.
    temperature = float(getattr(sampler, "temperature", 1.0))
    if isinstance(sampler, GreedySampler):
        kwargs = {"temperature": 0.0}
    elif isinstance(sampler, TopKSampler):
        kwargs = {"temperature": temperature, "top_k": int(sampler.k)}
    elif isinstance(sampler, TopPSampler):
        kwargs = {"temperature": temperature, "top_p": float(sampler.p)}
        if getattr(sampler, "k", None):
            kwargs["top_k"] = int(sampler.k)
    elif isinstance(sampler, RandomSampler):
        kwargs = {"temperature": temperature}
    else:
        logging.warning(
            "KerasHub sampler %s has no vLLM equivalent; "
            "falling back to vLLM's default sampling.",
            type(sampler).__name__,
        )
        return None
    seed = getattr(sampler, "seed", None)
    if seed is not None:
        kwargs["seed"] = int(seed)
    return kwargs


def _default_sampling_kwargs():
    """Maps KerasHub's default sampler to vLLM sampling kwargs.

    Every KerasHub task compiles itself with `CausalLM.compile`'s default
    sampler on construction, and presets don't serialize a custom one, so
    that default is what every preset would generate with. It is read from
    the `compile` signature instead of building the task: a JAX model build
    would allocate real parameter memory before vLLM's engine starts.
    Returns `None` if anything fails — serving then uses vLLM's defaults.
    """
    try:
        default = inspect.signature(CausalLM.compile).parameters["sampler"]
        return sampler_to_sampling_kwargs(get_sampler(default.default))
    except Exception as e:  # noqa: BLE001 - defaults are best-effort
        logging.warning(
            "Could not read the default sampler (%s); "
            "using vLLM's default sampling.",
            e,
        )
        return None


if _BaseLLM is not None:

    @keras_hub_export("keras_hub.vllm.KerasHubLLM")
    class KerasHubLLM(_BaseLLM):
        """A `vllm.LLM` that accepts the ``keras_hub:<preset>`` model scheme.

        A ``keras_hub:`` model string is materialized into a model directory
        (via `setup_vllm_model`) and tokenization is served by the preset's
        own KerasHub tokenizer (``tokenizer_mode="keras_hub"``) before
        delegating to ``vllm.LLM``. Any other model string is passed straight
        through, so the class behaves exactly like ``vllm.LLM`` for
        non-KerasHub models.

        When `generate` is called without `sampling_params`, the defaults
        come from the model's own compiled sampler (e.g. GPT-2's `top_k`),
        translated into `SamplingParams` — not from vLLM's generic defaults.
        Passing explicit `sampling_params` overrides this.

        Example::

            from keras_hub.vllm import KerasHubLLM
            llm = KerasHubLLM("keras_hub:gpt2_base_en")
            llm.generate(["The future of AI is"])  # model's own sampling
        """

        def __init__(self, model, **kwargs):
            self._default_sampling_kwargs = None
            # TemporaryDirectory backing a keras_hub preset's config dir.
            # Held on the instance so it lives as long as this LLM and is
            # deleted when the instance is garbage collected.
            self._keras_hub_dir = None
            is_keras_hub = isinstance(model, str) and model.startswith(
                "keras_hub:"
            )
            if is_keras_hub:
                preset = model.split("keras_hub:", 1)[1]
                # dtype defaults to bf16 (TPU paged KV cache). It is written
                # into config.json as torch_dtype *and* forwarded to vLLM.
                # Forwarding matters: left unset, vLLM resolves dtype="auto"
                # and downcasts a float32 config to bfloat16, while the
                # backbone would still build in float32 from torch_dtype --
                # the model would then hand float32 q/k/v to a bfloat16 paged
                # cache, which the attention kernel rejects.
                dtype = kwargs.pop("dtype", "bfloat16")
                kwargs.setdefault("dtype", dtype)
                # Use tpu-inference's native flax/nnx path, where the
                # KerasHubForCausalLM architecture is registered. This
                # change is process-wide on purpose: MODEL_IMPL_TYPE is an
                # environment knob that tpu-inference reads at model load,
                # in this process and in any engine workers it spawns, so
                # there is no narrower place to set it.
                os.environ.pop("MODEL_IMPL_TYPE", None)
                self._default_sampling_kwargs = _default_sampling_kwargs()
                self._keras_hub_dir = setup_vllm_model(
                    preset,
                    dtype=dtype,
                    max_model_len=kwargs.get("max_model_len"),
                )
                model = self._keras_hub_dir.name
                kwargs.setdefault("tokenizer", model)
                # Serve the preset's own KerasHub tokenizer through vLLM's
                # tokenizer registry (tokenizer_mode="keras_hub") — no HF
                # conversion. Registered by module path, so vLLM resolves
                # the class lazily in whichever process loads the tokenizer.
                TokenizerRegistry.register(
                    "keras_hub",
                    "keras_hub.src.vllm.tokenizer",
                    "KerasHubTokenizer",
                )
                # vLLM keys its prompt renderer by the same mode; the
                # standard HF renderer drives any TokenizerLike, so map
                # "keras_hub" to it (as vLLM does for e.g. kimi_audio).
                RENDERER_REGISTRY.register(
                    "keras_hub", "vllm.renderers.hf", "HfRenderer"
                )
                kwargs.setdefault("tokenizer_mode", "keras_hub")
            super().__init__(model=model, **kwargs)

        def generate(self, prompts=None, sampling_params=None, **kwargs):
            """Generates, defaulting to the model's own sampler settings.

            Text prompts pass through unchanged, so vLLM tokenizes them with
            the preset's own tokenizer (which prepends the start token for
            presets that use one) and `RequestOutput.prompt` stays populated.
            """
            if sampling_params is None and self._default_sampling_kwargs:
                sampling_params = SamplingParams(
                    **self._default_sampling_kwargs
                )
            return super().generate(
                prompts=prompts, sampling_params=sampling_params, **kwargs
            )

else:

    @keras_hub_export("keras_hub.vllm.KerasHubLLM")
    class KerasHubLLM:  # pragma: no cover - exercised only without vLLM
        """Placeholder when vLLM is unavailable; raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "KerasHubLLM requires vLLM with tokenizer-registry "
                "support. For TPU serving install version 0.24.0 or "
                "newer: `pip install 'vllm-tpu>=0.24.0'`."
            )
