"""
Registry module for Keras Hub to vLLM Integration.

Hooks into vLLM's model loading mechanism so that `LLM(model="keras_hub:preset_name")`
is recognized and routed to the `KerasVLLMAdapter`.
"""

import json
import logging
import os
import shutil
import tempfile

from keras_hub.src.vllm.adapter import KerasVLLMAdapter


def _verify_tokenizer_dir(temp_dir: str) -> bool:
    """Loads the exported tokenizer back and checks it tokenizes to non-empty.

    Guards against transformers/tokenizers versions that build an empty
    tokenizer from the exported assets (which is worse than no tokenizer).
    """
    from transformers import AutoTokenizer

    rt = AutoTokenizer.from_pretrained(temp_dir)
    return len(rt("hello world").get("input_ids", [])) > 0


def _export_sentencepiece(tokenizer, temp_dir: str, proto: bytes) -> bool:
    """Exports a KerasHub SentencePiece tokenizer (Gemma/Llama/Mistral) for HF.

    Writes the raw SP proto as `tokenizer.model` plus a tokenizer_config naming
    the matching HF class, then verifies it round-trips. On failure, removes the
    files so the caller falls back to skip_tokenizer_init + pre-tokenization.
    """
    try:
        with open(os.path.join(temp_dir, "tokenizer.model"), "wb") as f:
            f.write(proto)
    except Exception as e:  # noqa: BLE001
        logging.warning("Could not write SentencePiece tokenizer.model: %s", e)
        return False

    cls_name = type(tokenizer).__name__.lower()
    hf_class = "GemmaTokenizer" if "gemma" in cls_name else "LlamaTokenizer"
    with open(
        os.path.join(temp_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "tokenizer_class": hf_class,
                "legacy": False,
                "add_bos_token": True,
                "add_eos_token": False,
            },
            f,
        )

    try:
        if _verify_tokenizer_dir(temp_dir):
            return True
        raise ValueError("exported SentencePiece tokenizer produced empty output")
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "SentencePiece tokenizer unusable (%s); use skip_tokenizer_init=True "
            "+ pre-tokenized input for this preset.",
            e,
        )
        for fn in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json"):
            p = os.path.join(temp_dir, fn)
            if os.path.exists(p):
                os.remove(p)
        return False


def _export_hf_tokenizer(tokenizer, temp_dir: str) -> bool:
    """Writes HF-compatible tokenizer files so vLLM can tokenize raw text.

    Handles KerasHub's byte-level BPE tokenizers (GPT-2, OPT, ... -> stock
    `GPT2Tokenizer`) and SentencePiece tokenizers (Gemma, Llama, Mistral ->
    `tokenizer.model` + Llama/Gemma tokenizer). Returns True on success; on
    failure the caller should use ``skip_tokenizer_init=True`` + pre-tokenized
    input.
    """
    # SentencePiece family: KerasHub stores the serialized proto on `.proto`.
    proto = getattr(tokenizer, "proto", None)
    if proto:
        return _export_sentencepiece(tokenizer, temp_dir, proto)

    # Byte-level BPE family.
    if not (hasattr(tokenizer, "merges") and hasattr(tokenizer, "save_assets")):
        return False
    try:
        # Writes vocabulary.json + merges.txt into temp_dir.
        tokenizer.save_assets(temp_dir)
    except Exception as e:  # noqa: BLE001
        logging.warning("Tokenizer save_assets failed: %s", e)
        return False

    vocab_src = os.path.join(temp_dir, "vocabulary.json")
    vocab_dst = os.path.join(temp_dir, "vocab.json")  # HF expects this name
    if os.path.exists(vocab_src) and not os.path.exists(vocab_dst):
        shutil.copyfile(vocab_src, vocab_dst)
    merges_path = os.path.join(temp_dir, "merges.txt")
    if not (os.path.exists(vocab_dst) and os.path.exists(merges_path)):
        return False

    # HF's GPT2Tokenizer drops the first line of merges.txt (expects a version
    # header). KerasHub writes no header, so prepend one or the first real merge
    # rule would be silently lost.
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_content = f.read()
    if not merges_content.startswith("#version"):
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n" + merges_content)

    eos = "<|endoftext|>"

    # Prefer emitting a fast tokenizer (tokenizer.json), which vLLM v1's
    # incremental detokenizer favors. Building GPT2TokenizerFast from the
    # vocab/merges and calling save_pretrained writes tokenizer.json plus
    # consistent tokenizer_config.json / special_tokens_map.json.
    try:
        from transformers import AutoTokenizer, GPT2TokenizerFast

        fast = GPT2TokenizerFast(
            vocab_file=vocab_dst,
            merges_file=merges_path,
            unk_token=eos,
            bos_token=eos,
            eos_token=eos,
            add_prefix_space=False,
        )
        fast.save_pretrained(temp_dir)
        # Verify the emitted tokenizer.json actually round-trips (some
        # transformers/tokenizers versions build an empty byte-level BPE from
        # vocab+merges). If it tokenizes to nothing, it's worse than the slow
        # tokenizer — remove it and fall back.
        reloaded = AutoTokenizer.from_pretrained(temp_dir)
        if len(reloaded("hello world").get("input_ids", [])) > 0:
            return True
        raise ValueError("emitted fast tokenizer produced empty output")
    except Exception as e:  # noqa: BLE001 - fall back to the slow tokenizer
        logging.warning(
            "Fast tokenizer.json unusable (%s); using slow GPT2Tokenizer.", e
        )
        # Remove a possibly-broken tokenizer.json so vLLM loads the slow files
        # (vocab.json + merges.txt) instead of the bad fast tokenizer.
        broken = os.path.join(temp_dir, "tokenizer.json")
        if os.path.exists(broken):
            os.remove(broken)

    # Slow-tokenizer fallback: minimal config over vocab.json + merges.txt.
    with open(
        os.path.join(temp_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "tokenizer_class": "GPT2Tokenizer",
                "bos_token": eos,
                "eos_token": eos,
                "unk_token": eos,
                "add_prefix_space": False,
                "clean_up_tokenization_spaces": False,
            },
            f,
        )
    with open(
        os.path.join(temp_dir, "special_tokens_map.json"), "w", encoding="utf-8"
    ) as f:
        json.dump({"bos_token": eos, "eos_token": eos, "unk_token": eos}, f)
    return True


def _copy_preset_tokenizer_json(preset: str, temp_dir: str) -> bool:
    """Fallback: copy the preset's own HF ``tokenizer.json`` into ``temp_dir``.

    Some presets (e.g. ``gpt2_large_en``) ship a ready-made fast-tokenizer
    ``tokenizer.json`` but omit the ``vocabulary.json`` that KerasHub's
    ``Tokenizer.from_preset`` needs, so the normal export path can't run. vLLM
    loads ``tokenizer.json`` directly, so this keeps raw-text prompts working.
    Returns True only if the copied tokenizer round-trips.
    """
    try:
        from keras_hub.src.utils import preset_utils

        src = preset_utils.get_file(preset, "tokenizer.json")
    except Exception as e:  # noqa: BLE001
        logging.warning("No preset tokenizer.json for %r: %s", preset, e)
        return False
    if not src or not os.path.exists(src):
        return False
    shutil.copyfile(src, os.path.join(temp_dir, "tokenizer.json"))
    cfg_path = os.path.join(temp_dir, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "clean_up_tokenization_spaces": False,
                },
                f,
            )
    try:
        from transformers import AutoTokenizer

        reloaded = AutoTokenizer.from_pretrained(temp_dir)
        if len(reloaded("hello world").get("input_ids", [])) > 0:
            return True
        logging.warning("Preset tokenizer.json for %r produced empty output.", preset)
    except Exception as e:  # noqa: BLE001
        logging.warning("Preset tokenizer.json unusable for %r: %s", preset, e)
    return False


def _register_model_architecture() -> None:
    """Registers KerasVLLMAdapter with vLLM's internal model registry."""
    try:
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.register_model("KerasVLLMAdapter", KerasVLLMAdapter)
    except ImportError:
        logging.warning(
            "Skipping KerasVLLMAdapter registration. vLLM is not installed "
            "or the ModelRegistry module could not be imported."
        )


def register_keras_hub_models() -> None:
    """Registers KerasVLLMAdapter with vLLM via its sanctioned ModelRegistry.

    No monkeypatching: architecture recognition goes through vLLM's public
    `ModelRegistry.register_model` here, and through tpu-inference's native
    `_get_model_architecture` hook on the serving side. When `setup_vllm_model`
    materializes a model directory, vLLM loads the `KerasVLLMAdapter`.
    """
    _register_model_architecture()


def setup_vllm_model(preset: str, dtype: str = "float16") -> str:
    """Creates a configuration directory for vLLM to load a Keras Hub preset.

    Args:
        preset: The Keras Hub preset name (e.g., "gemma_2b_en").
        dtype: The torch dtype to run inference with.

    Returns:
        The path to the temporary configuration directory to pass to `vllm.LLM`.
    """
    temp_dir = tempfile.mkdtemp(prefix="keras_hub_vllm_")

    preset_lower = preset.lower()
    if "opt" in preset_lower:
        model_type = "opt"
    elif "gpt2" in preset_lower or "gpt_2" in preset_lower:
        model_type = "gpt2"
    elif "gemma" in preset_lower:
        model_type = "gemma2"
    else:
        model_type = "gemma2"

    # Derive the true vocabulary size from the preset's tokenizer. vLLM relies
    # on vocab_size for KV-cache memory profiling, so a hardcoded value
    # silently mis-profiles any model other than the one it was tuned for
    # (e.g. GPT-2's 50257 vs. Gemma's 256000).
    vocab_size = None
    eos_token_id = None
    tokenizer = None
    # Load the KerasHub tokenizer (best effort) for vocab_size / eos and as the
    # primary HF-export source.
    try:
        from keras_hub import models

        tokenizer = models.Tokenizer.from_preset(preset)
        vocab_size = int(tokenizer.vocabulary_size())
        eos_token_id = getattr(tokenizer, "end_token_id", None)
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "Could not load KerasHub tokenizer for preset %r (%s).", preset, e
        )

    # Export an HF tokenizer so vLLM accepts raw text. Prefer the KerasHub
    # tokenizer; if that's unavailable/fails, fall back to the preset's own
    # tokenizer.json (some presets ship it without vocabulary.json).
    exported = False
    if tokenizer is not None:
        try:
            exported = _export_hf_tokenizer(tokenizer, temp_dir)
        except Exception as e:  # noqa: BLE001
            logging.warning("HF tokenizer export failed for %r: %s", preset, e)
    if not exported:
        exported = _copy_preset_tokenizer_json(preset, temp_dir)
    if exported:
        logging.info("Exported HF tokenizer assets for preset %r.", preset)
    else:
        logging.warning(
            "Could not export an HF tokenizer for preset %r; use "
            "skip_tokenizer_init=True + pre-tokenized input.",
            preset,
        )

    # Sane vocab_size default by family when it couldn't be inferred (vLLM uses
    # it for KV-cache memory profiling, so the gemma default mis-sizes GPT-2/OPT).
    if vocab_size is None:
        if "gpt2" in preset_lower or "gpt_2" in preset_lower:
            vocab_size = 50257
        elif "opt" in preset_lower:
            vocab_size = 50272
        else:
            vocab_size = 256000
        logging.warning(
            "vocab_size not inferred for %r; defaulting to %d.", preset, vocab_size
        )

    config_dict = {
        "architectures": ["KerasVLLMAdapter"],
        "_name_or_path": f"keras_hub:{preset}",
        "keras_hub_preset": preset,
        "torch_dtype": dtype,
        "model_type": model_type,
        "vocab_size": vocab_size,
    }
    if eos_token_id is not None:
        config_dict["eos_token_id"] = int(eos_token_id)
        config_dict["bos_token_id"] = int(
            getattr(tokenizer, "start_token_id", None) or eos_token_id
        )

    # Write the REAL architecture dims so vLLM allocates a matching KV cache.
    # Without these, vLLM guesses from `model_type` (e.g. "gemma2" -> HF
    # Gemma-2 defaults), which mismatches the actual KerasHub backbone and fails
    # the kernel's kv_cache-shape check (num_kv_heads / num_layers / head_dim).
    config_dict.update(_derive_arch_config(preset))

    with open(os.path.join(temp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f)

    return temp_dir


def _derive_arch_config(preset: str) -> dict:
    """Reads the KerasHub backbone config and maps it to HF/vLLM config keys.

    Reads the preset's serialized ``config.json`` directly (no model build, no
    TPU allocation) and translates its backbone args to the fields vLLM uses for
    KV-cache allocation. Works across families: GPT-2 uses ``num_heads`` (no
    GQA); Gemma/Llama/Mistral use ``num_query_heads`` / ``num_key_value_heads``
    / ``head_dim``.
    """
    try:
        from keras_hub.src.utils import preset_utils

        cfg = preset_utils.load_json(preset).get("config", {})
    except Exception as e:  # noqa: BLE001
        logging.warning(
            "Could not derive architecture for %r (%s); vLLM will guess from "
            "model_type, which may mismatch the KV cache.",
            preset,
            e,
        )
        return {}

    hidden = cfg.get("hidden_dim")
    n_heads = cfg.get("num_query_heads") or cfg.get("num_heads")
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
    return arch


# vLLM is optional at import time, so the subclass is defined only when present.
try:
    from vllm import LLM as _BaseLLM
except Exception:  # noqa: BLE001 - vllm not installed / import failure
    _BaseLLM = None


if _BaseLLM is not None:

    class KerasHubLLM(_BaseLLM):
        """A `vllm.LLM` that accepts the ``keras_hub:<preset>`` model scheme.

        This is plain subclassing (no monkeypatching): a ``keras_hub:`` model
        string is normalized into a materialized model directory (config +
        tokenizer) before delegating to ``vllm.LLM``. Any other model string is
        passed straight through, so the class behaves exactly like ``vllm.LLM``
        for non-KerasHub models.

        Example::

            from keras_hub.src.vllm import KerasHubLLM
            llm = KerasHubLLM("keras_hub:gpt2_base_en")
            llm.generate(["The future of AI is"], sampling_params)
        """

        def __init__(self, model, **kwargs):
            if isinstance(model, str) and model.startswith("keras_hub:"):
                preset = model.split("keras_hub:", 1)[1]
                # dtype defaults to bf16 (TPU paged KV cache); consumed here, not
                # forwarded — the exported config.json carries torch_dtype.
                dtype = kwargs.pop("dtype", "bfloat16")
                os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
                register_keras_hub_models()
                model = setup_vllm_model(preset, dtype=dtype)
                kwargs.setdefault("tokenizer", model)
            super().__init__(model=model, **kwargs)

else:

    class KerasHubLLM:  # pragma: no cover - exercised only without vLLM
        """Placeholder when vLLM is unavailable; raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "KerasHubLLM requires vLLM. Install vllm (or vllm-tpu) first."
            )
