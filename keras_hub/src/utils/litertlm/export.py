"""Export KerasHub CausalLM models to LiteRT-LM `.litertlm` bundles."""

import contextlib
import dataclasses
import importlib.util
import json
import os
import tempfile
import warnings

import keras

try:
    import torch
except ImportError:
    torch = None

from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)
from keras_hub.src.utils.litertlm.hf_tokenizer_converter import (
    materialize_hf_tokenizer_json,
)
from keras_hub.src.utils.litertlm.model_specs import SamplerConfig
from keras_hub.src.utils.litertlm.model_specs import resolve_export_spec
from keras_hub.src.utils.preset_utils import TOKENIZER_ASSET_DIR

# ``litert_torch`` is an optional dependency. Use ``find_spec`` to check for
# availability without importing it at module level, because importing
# ``litert_torch`` has the side effect of enabling ``jax_enable_x64``.
_LITERT_TORCH_AVAILABLE = importlib.util.find_spec("litert_torch") is not None


@contextlib.contextmanager
def _preserve_jax_config_state(name, value=None):
    """Preserve the JAX config flag ``name`` around ``litert_torch`` usage.

    When ``value`` is given, the flag is pinned to it for the block.
    """
    try:
        import jax
    except ImportError:
        jax = None
        original = None
    else:
        original = getattr(jax.config, name)
    if jax is not None and value is not None:
        jax.config.update(name, value)
    try:
        yield
    finally:
        if jax is not None:
            jax.config.update(name, original)


# ``torch`` is optional. Defining the adapter bases conditionally lets the
# module import cleanly in non-PyTorch environments while still giving
# ``torch.nn.Module`` semantics when PyTorch is present.
if torch is not None:
    _AdapterBase = torch.nn.Module
else:
    _AdapterBase = object


class _PrefillAdapter(_AdapterBase):
    """Thin wrapper that exposes ``adapter.forward_prefill`` to litert_torch."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, *args, **kwargs):
        return self.base.forward_prefill(*args, **kwargs)


class _DecodeAdapter(_AdapterBase):
    """Thin wrapper that exposes ``adapter.forward_decode`` to litert_torch."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, *args, **kwargs):
        return self.base.forward_decode(*args, **kwargs)


# A cheap sanity check on `hf_tokenizer_path` compatibility, not exact
# validation: only a large absolute difference AND a >=5x ratio together
# flag a tokenizer from an entirely different model/family.
_HF_TOKENIZER_VOCAB_MISMATCH_ABS_THRESHOLD = 300
_HF_TOKENIZER_VOCAB_MISMATCH_RATIO_THRESHOLD = 5.0


def _model_embedding_vocab_size(model):
    """Return the model's embedding vocabulary size, or ``None`` if unknown.

    Prefers ``backbone.vocabulary_size`` (the constructor argument most
    backbones store directly, e.g. ``GemmaBackbone``/``LlamaBackbone``/
    ``GPT2Backbone``); falls back to ``backbone.token_embedding.input_dim``
    (the actual embedding table size) for backbones that do not expose
    ``vocabulary_size`` directly.
    """
    backbone = getattr(model, "backbone", None)
    vocab_size = getattr(backbone, "vocabulary_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    token_embedding = getattr(backbone, "token_embedding", None)
    input_dim = getattr(token_embedding, "input_dim", None)
    if input_dim is not None:
        return int(input_dim)
    return None


def _hf_tokenizer_vocab_size(hf_tokenizer_path):
    """Return the vocab size implied by a HuggingFace ``tokenizer.json``.

    Reads the file directly as JSON (``tokenizer.json`` is plain JSON; this
    avoids a hard dependency on the ``tokenizers`` library just to sanity
    check a vocab size) and returns ``max_token_id + 1`` across both the
    base ``model.vocab`` mapping and any ``added_tokens`` entries (special
    tokens are often listed separately from the base vocab) -- matching how
    large the embedding table must be to cover every id the tokenizer can
    produce. Returns ``None`` if the file cannot be parsed as the expected
    ``tokenizer.json`` structure.
    """
    try:
        with open(hf_tokenizer_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    model_vocab = (data.get("model") or {}).get("vocab") or {}
    try:
        max_id = max(model_vocab.values(), default=-1)
    except (TypeError, ValueError):
        return None
    for token in data.get("added_tokens") or []:
        token_id = token.get("id")
        if isinstance(token_id, int):
            max_id = max(max_id, token_id)
    if max_id < 0:
        return None
    return max_id + 1


def _check_hf_tokenizer_vocab_compatible(hf_tokenizer_path, model):
    """Raise ``ValueError`` if the HF tokenizer's vocab looks incompatible.

    This is a cheap sanity check (see the module-level threshold constants
    above), not exact validation -- it exists to catch the case of bundling
    a tokenizer from an entirely different model/family, not to enforce
    that the tokenizer and model agree token-for-token.
    """
    hf_vocab_size = _hf_tokenizer_vocab_size(hf_tokenizer_path)
    model_vocab_size = _model_embedding_vocab_size(model)
    if hf_vocab_size is None or not model_vocab_size:
        # Could not determine one of the two sizes; skip rather than risk a
        # false positive from an unusual tokenizer.json structure or backbone.
        return
    diff = abs(hf_vocab_size - model_vocab_size)
    ratio = hf_vocab_size / model_vocab_size
    is_grossly_mismatched = (
        diff > _HF_TOKENIZER_VOCAB_MISMATCH_ABS_THRESHOLD
        and (
            ratio >= _HF_TOKENIZER_VOCAB_MISMATCH_RATIO_THRESHOLD
            or ratio <= 1 / _HF_TOKENIZER_VOCAB_MISMATCH_RATIO_THRESHOLD
        )
    )
    if is_grossly_mismatched:
        raise ValueError(
            "`hf_tokenizer_path` appears incompatible with the model: the "
            f"tokenizer implies a vocabulary of {hf_vocab_size} tokens "
            f"(highest token id + 1 across `model.vocab` and "
            f"`added_tokens` in {hf_tokenizer_path!r}), but the model's "
            f"embedding table is sized for {model_vocab_size} tokens "
            f"(`{type(model.backbone).__name__}`). This looks like a "
            "tokenizer from a different model/family rather than a small "
            "reserved-token discrepancy -- pass the tokenizer that matches "
            "this model, or omit `hf_tokenizer_path` to use the model's own "
            "tokenizer."
        )


def _validate_export_args(
    model,
    path,
    tokenizer,
    backend_constraint,
    hf_tokenizer_path,
    prefill_seq_len,
):
    """Fail fast on invalid export arguments.

    Importing ``litert_torch`` is deferred to the orchestrator so that the
    JAX ``jax_enable_x64`` side effect can be kept under one
    preserve/restore context that covers both import and tracing.

    Returns:
        A ``(prefill_seq_lens, backend_constraint)`` tuple: the normalized
        list of prefill sequence lengths, and the normalized (lowercased)
        ``backend_constraint`` string (or ``None``). Callers must use the
        returned ``backend_constraint`` -- not the original argument -- so
        the lowercased value actually reaches ``_assemble_bundle`` /
        ``builder.add_tflite_model``.
    """
    if not path.endswith(".litertlm"):
        raise ValueError(
            "LiteRT-LM export requires a filepath ending in `.litertlm`. "
            f"Received: path={path}"
        )

    if not hasattr(model, "call_with_cache"):
        raise ValueError(
            "LiteRT-LM export requires a model with a `call_with_cache()` "
            "method."
        )

    if backend_constraint is not None:
        if not isinstance(backend_constraint, str):
            raise ValueError(
                "`backend_constraint` must be a string or None. "
                f"Received: {backend_constraint!r}"
            )
        backend_constraint = backend_constraint.lower()

    if hf_tokenizer_path is not None:
        hf_tokenizer_path = os.fspath(hf_tokenizer_path)
        if not os.path.isfile(hf_tokenizer_path):
            raise ValueError(
                "`hf_tokenizer_path` must point to an existing file. "
                f"Received: {hf_tokenizer_path!r}"
            )
        if not hf_tokenizer_path.endswith(".json"):
            raise ValueError(
                "`hf_tokenizer_path` must point to a `tokenizer.json` file. "
                f"Received: {hf_tokenizer_path!r}"
            )
        _check_hf_tokenizer_vocab_compatible(hf_tokenizer_path, model)
    # Any BytePairTokenizer subclass can be converted to HF tokenizer.json.
    elif not _is_sentencepiece_tokenizer(tokenizer) and not isinstance(
        tokenizer, BytePairTokenizer
    ):
        raise ValueError(
            "LiteRT-LM export supports SentencePiece tokenizers and any "
            "BytePairTokenizer subclass. Received: "
            f"{type(tokenizer).__module__}.{type(tokenizer).__name__}."
        )

    # PyTorch is required for tracing and for building sample inputs. Surface
    # this before the backend check so a JAX/TF caller without torch installed
    dtype = _torch_dtype_from_model(model)

    # Bundle all resolved per-export-run settings into one immutable plan.
    plan = ExportPlan(
        spec=spec,
        num_layers=num_layers,
        cache_length=cache_length,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        prefill_seq_lens=prefill_seq_lens,
        dtype=dtype,
        sampler_config=sampler_config,
        model_type_overridden=llm_model_type is not None,
    )

    with _cpu_default_device_scope():
        prefill_inputs_map = _build_prefill_inputs(plan)

        decode_inputs = _build_sample_inputs(
            batch_size=1,
            seq_len=1,
            num_layers=plan.num_layers,
            cache_length=plan.cache_length,
            num_kv_heads=plan.num_kv_heads,
            head_dim=plan.head_dim,
            dtype=plan.dtype,
            spec=plan.spec,
        )

        adapter = KerasHubLiteRTAdapter(
            model,
            plan.num_layers,
            plan.cache_length,
            export_spec=spec,
        )
        adapter.eval()

        prefill_adapter = _PrefillAdapter(adapter).eval()
        decode_adapter = _DecodeAdapter(adapter).eval()

        # The JAX bridge defaults to TPU when one is visible; force CPU so
        # export does not contend with other processes using the TPU.
        with (
            _preserve_jax_config_state("jax_enable_x64"),
            _preserve_jax_config_state("jax_platforms", "cpu"),
        ):
            import litert_torch

            # The import above enables ``jax_enable_x64`` only on first
            # import; the JAX bridge requires x64 for consistent int64
            # dtypes, so pin it explicitly for the conversion.
            try:
                import jax

                jax.config.update("jax_enable_x64", True)
            except ImportError:
                pass

            edge_model = _trace_and_convert(
                litert_torch,
                prefill_adapter,
                decode_adapter,
                prefill_inputs_map,
                decode_inputs,
                plan,
                **kwargs,
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        _assemble_bundle(
            path=path,
            temp_dir=temp_dir,
            tokenizer=tokenizer,
            backend_constraint=backend_constraint,
            edge_model=edge_model,
            plan=plan,
            hf_tokenizer_path=hf_tokenizer_path,
        )

    return path


def _build_sample_inputs(
    batch_size,
    seq_len,
    num_layers,
    cache_length,
    num_kv_heads,
    head_dim,
    dtype,
    spec,
):
    """Create concrete sample tensors for one signature.

    The per-layer KV-cache sample shape is owned by the model family's
    spec (``spec.get_kv_cache_sample_shape``; see ``cache_layout`` on
    ``LiteRTLMExportSpec``).
    """
    device = "cpu"
    tokens = torch.zeros(
        (batch_size, seq_len), dtype=torch.int32, device=device
    )
    input_pos = torch.arange(seq_len, dtype=torch.int32, device=device)
    if seq_len == 1:
        input_pos = torch.zeros((1,), dtype=torch.int32, device=device)
    kv_cache = {}
    shape = spec.get_kv_cache_sample_shape(
        batch_size, cache_length, num_kv_heads, head_dim
    )
    for i in range(num_layers):
        kv_cache[f"kv_cache_k_{i}"] = torch.zeros(
            shape, dtype=dtype, device=device
        )
        kv_cache[f"kv_cache_v_{i}"] = torch.zeros(
            shape, dtype=dtype, device=device
        )

    sample = {
        "tokens": tokens,
        "input_pos": input_pos,
    }
    sample.update(kv_cache)
    return sample


def _get_tokenizer(model):
    preprocessor = getattr(model, "preprocessor", None)
    if preprocessor is None:
        raise ValueError(
            "LiteRT-LM export requires an attached preprocessor with a "
            "tokenizer."
        )
    tokenizer = getattr(preprocessor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "LiteRT-LM export requires an attached tokenizer on the "
            "preprocessor."
        )
    return tokenizer


def _is_sentencepiece_tokenizer(tokenizer):
    """Return ``True`` if *tokenizer* is SentencePiece-compatible."""
    if isinstance(tokenizer, SentencePieceTokenizer):
        return True
    file_assets = set(getattr(tokenizer, "file_assets", []) or [])
    return "vocabulary.spm" in file_assets


def _materialize_sentencepiece_tokenizer(tokenizer, temp_dir):
    preset_dir = os.path.join(temp_dir, "tokenizer_preset")
    tokenizer.save_to_preset(preset_dir)
    tokenizer_path = os.path.join(
        preset_dir, TOKENIZER_ASSET_DIR, "vocabulary.spm"
    )
    if not os.path.exists(tokenizer_path):
        raise ValueError(
            "Failed to materialize the SentencePiece tokenizer asset at "
            f"{tokenizer_path}."
        )
    return tokenizer_path


def _build_llm_metadata(
    spec,
    tokenizer,
    max_num_tokens,
    path,
    sampler_config=None,
    model_type_overridden=False,
):
    """Serialize an ``LlmMetadata`` protobuf to *path*."""
    # The protobuf lives under an internal-looking subpackage of
    # ``litert-lm-builder``; import defensively and surface a clear error
    # if the internal layout changes.
    try:
        from litert_lm_builder.runtime.proto import llm_metadata_pb2
    except ImportError as e:
        raise ImportError(
            "LiteRT-LM export requires the metadata protobuf from "
            "`litert-lm-builder`. The internal module layout appears to have "
            "changed. Please verify your `litert-lm-builder` installation."
        ) from e

    meta = llm_metadata_pb2.LlmMetadata()

    start_id = getattr(tokenizer, "start_token_id", None)
    if start_id is not None:
        meta.start_token.token_ids.ids.append(int(start_id))

    # The primary EOS (used for packing/training) is always a stop token.
    stop_token_ids = []
    end_id = getattr(tokenizer, "end_token_id", None)
    if end_id is not None:
        stop_token_ids.append(int(end_id))

    # Add each family's extra chat-turn stop token (e.g. Gemma's
    # `<end_of_turn>`; see `LiteRTLMExportSpec.get_chat_stop_token_ids`),
    # de-duplicated against `end_token_id`.
    for extra_id in spec.get_chat_stop_token_ids(tokenizer):
        extra_id = int(extra_id)
        if extra_id not in stop_token_ids:
            stop_token_ids.append(extra_id)

    for stop_id in stop_token_ids:
        meta.stop_tokens.add().token_ids.ids.append(stop_id)

    meta.max_num_tokens = int(max_num_tokens)

    getattr(meta.llm_model_type, spec.model_type).SetInParent()

    # Populate function-calling fields (only `FunctionGemmaSpec` overrides
    # the base no-op). Skipped on an explicit `llm_model_type` override:
    # litert-torch also skips its model-specific metadata builder then.
    if not model_type_overridden:
        spec.populate_function_gemma_metadata(meta)

    # Sampler defaults are written only when the caller passes a
    # `sampler_config` (mirroring litert-torch's conditional
    # `sampler_params`); otherwise the runtime picks its own policy.
    if sampler_config is not None:
        try:
            from litert_lm_builder.runtime.proto import sampler_params_pb2
        except ImportError as e:
            raise ImportError(
                "LiteRT-LM export requires the sampler protobuf from "
                "`litert-lm-builder`. The internal module layout appears to "
                "have changed. Please verify your `litert-lm-builder` "
                "installation."
            ) from e

        sp = meta.sampler_params
        top_k = sampler_config.top_k
        if top_k is not None:
            sp.k = top_k
        if sampler_config.top_p is not None:
            sp.p = sampler_config.top_p
        if sampler_config.temperature is not None:
            sp.temperature = sampler_config.temperature
        if sampler_config.seed is not None:
            sp.seed = sampler_config.seed

        SamplerParameters = sampler_params_pb2.SamplerParameters
        # `GREEDY` (type 3) is not implemented by litertlm-android 0.13.1 or
        # the host Python `litert_lm` 0.13.1 runtime. Emit `TOP_K` with k=1
        # instead, which is functionally equivalent and is implemented.
        if top_k is not None:
            sp.type = SamplerParameters.TOP_K
        else:
            sp.type = SamplerParameters.TOP_P

    with open(path, "wb") as f:
        f.write(meta.SerializeToString())


def _torch_dtype_from_model(model):
    """Return a ``torch.dtype`` matching the model's compute dtype."""
    from keras.src.backend.torch import core as torch_core

    compute_dtype = getattr(model, "compute_dtype", None)
    if compute_dtype is None:
        compute_dtype = getattr(model.backbone, "compute_dtype", "float32")
    # Unwrap DTypePolicy first: `to_torch_dtype` only accepts hashable dtypes.
    if hasattr(compute_dtype, "name"):
        compute_dtype = compute_dtype.name
    if not isinstance(compute_dtype, (str, torch.dtype)):
        raise ValueError(
            "The model's `compute_dtype` must be a dtype string, a "
            "`torch.dtype`, or a Keras `DTypePolicy` for LiteRT-LM export. "
            f"Received: compute_dtype={compute_dtype!r}"
        )
    try:
        torch_dtype = torch_core.to_torch_dtype(compute_dtype)
    except ValueError as e:
        raise ValueError(
            "The model's `compute_dtype` must map to a PyTorch dtype for "
            f"LiteRT-LM export. Received: compute_dtype={compute_dtype!r}"
        ) from e
    if torch_dtype is torch.bfloat16:
        warnings.warn(
            "Exporting with `compute_dtype=bfloat16`. BF16 LiteRT-LM export "
            "is untested; numeric parity with the Keras model and runtime "
            "support are not guaranteed. Consider using float32 unless you "
            "have independently verified BF16 export for this model.",
            stacklevel=2,
        )
    return torch_dtype


def _import_litert_lm_builder():
    try:
        import litert_lm_builder
    except ImportError as e:
        raise ImportError(
            "LiteRT-LM export requires `litert-lm-builder`. "
            "Install it with: pip install litert-lm-builder"
        ) from e
    return litert_lm_builder
