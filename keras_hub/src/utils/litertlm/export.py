"""Export KerasHub CausalLM models to LiteRT-LM `.litertlm` bundles."""

import contextlib
import dataclasses
import importlib.util
import os
import tempfile
import warnings

import keras

try:
    import torch
except ImportError:
    torch = None

from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
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


def _validate_export_args(
    model,
    path,
    tokenizer,
    backend_constraint,
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

    if not _is_sentencepiece_tokenizer(tokenizer):
        raise ValueError(
            "LiteRT-LM export supports SentencePiece tokenizers. Received: "
            f"{type(tokenizer).__module__}.{type(tokenizer).__name__}."
        )

    # PyTorch is required for tracing and for building sample inputs. Surface
    # this before the backend check so a JAX/TF caller without torch installed
    # gets a clear message instead of a raw ``ModuleNotFoundError``.
    if torch is None:
        raise ImportError(
            "LiteRT-LM export requires PyTorch. "
            "Install it with: pip install torch"
        )

    # litert_torch only supports the PyTorch Keras backend.
    if keras.config.backend() != "torch":
        raise ValueError(
            "LiteRT-LM export is only supported with the PyTorch backend. "
            f"Current backend: {keras.config.backend()}."
        )

    # Now that tokenizer and backend checks are done, require the optional
    # litert-torch/litert-lm-builder packages for the actual export.
    if not _LITERT_TORCH_AVAILABLE:
        raise ImportError(
            "LiteRT-LM export requires `litert-torch`. "
            "Install it with: pip install litert-torch"
        )

    # Validate `backend_constraint` against the builder's enum only after the
    # availability checks above, so under-supported environments get the
    # friendly torch/backend/dependency errors first. Allowed values track
    # `litert_lm_builder.Backend`. The builder also accepts comma-separated
    # lists; KerasHub deliberately accepts a single backend only.
    if backend_constraint is not None:
        litert_lm_builder = _import_litert_lm_builder()
        valid_backends = {b.value for b in litert_lm_builder.Backend}
        if backend_constraint not in valid_backends:
            raise ValueError(
                f"Invalid backend_constraint: {backend_constraint!r}. "
                f"Must be one of {sorted(valid_backends)}."
            )

    # Normalise prefill_seq_len to a sorted list. Cache-length checks are left
    # to the orchestrator because ``cache_length`` is not known until after
    # ``spec.get_cache_config`` runs.
    if prefill_seq_len is None:
        prefill_seq_lens = None
    elif isinstance(prefill_seq_len, int):
        prefill_seq_lens = [prefill_seq_len]
    elif isinstance(prefill_seq_len, (list, tuple)):
        if not prefill_seq_len:
            raise ValueError("`prefill_seq_len` cannot be an empty list.")
        prefill_seq_lens = sorted(set(prefill_seq_len))
    else:
        raise ValueError(
            "`prefill_seq_len` must be an int or a list of ints. "
            f"Received: {prefill_seq_len!r}"
        )

    if prefill_seq_lens is not None:
        for seq_len in prefill_seq_lens:
            if not isinstance(seq_len, int) or seq_len <= 0:
                raise ValueError(
                    "`prefill_seq_len` values must be positive integers. "
                    f"Received: {seq_len!r}"
                )

    return prefill_seq_lens, backend_constraint


@dataclasses.dataclass(frozen=True)
class ExportPlan:
    """Immutable bundle of per-export-run settings for a single export call.

    ``export_to_litertlm`` computes all of these values once, early in the
    pipeline (resolving the model-family spec and cache config), then passes
    a single ``ExportPlan`` to the later pipeline phases (building sample
    inputs, tracing/converting, assembling the bundle) instead of a long,
    order-sensitive positional-argument list.
    """

    spec: object
    num_layers: int
    cache_length: int
    num_kv_heads: int
    head_dim: int
    prefill_seq_lens: list
    dtype: object
    sampler_config: object | None
    model_type_overridden: bool


def _build_prefill_inputs(plan):
    """Build a ``{seq_len: sample_inputs}`` map for every prefill bucket."""
    prefill_inputs_map = {}
    for seq_len in plan.prefill_seq_lens:
        prefill_inputs_map[seq_len] = _build_sample_inputs(
            batch_size=1,
            seq_len=seq_len,
            num_layers=plan.num_layers,
            cache_length=plan.cache_length,
            num_kv_heads=plan.num_kv_heads,
            head_dim=plan.head_dim,
            dtype=plan.dtype,
            spec=plan.spec,
        )
    return prefill_inputs_map


def _chain_signatures(litert_torch, signatures, **kwargs):
    """Chain multiple litert_torch signatures, hiding first/rest asymmetry."""
    converter = None
    for sig_name, adapter, sample_kwargs in signatures:
        if converter is None:
            converter = litert_torch.signature(
                sig_name,
                adapter,
                sample_kwargs=sample_kwargs,
                **kwargs,
            )
        else:
            converter = converter.signature(
                sig_name,
                adapter,
                sample_kwargs=sample_kwargs,
                **kwargs,
            )
    return converter


def _trace_and_convert(
    litert_torch,
    prefill_adapter,
    decode_adapter,
    prefill_inputs_map,
    decode_inputs,
    plan,
    **kwargs,
):
    """Trace the prefill/decode signatures and convert them to LiteRT."""
    # Defer torch-specific imports until the backend has been verified as
    # torch, so that non-torch callers get the friendly backend error.
    from keras_hub.src.utils.litertlm.traceable_ops import traceable_ops_scope

    with traceable_ops_scope():
        # Chain one signature per prefill bucket plus the decode signature.
        signatures = []
        for seq_len in plan.prefill_seq_lens:
            sig_name = (
                "prefill"
                if len(plan.prefill_seq_lens) == 1
                else f"prefill_{seq_len}"
            )
            signatures.append(
                (sig_name, prefill_adapter, prefill_inputs_map[seq_len])
            )
        signatures.append(("decode", decode_adapter, decode_inputs))

        converter = _chain_signatures(litert_torch, signatures, **kwargs)
        edge_model = converter.convert(lightweight_conversion=False)

    return edge_model


def _assemble_bundle(
    *,
    path,
    temp_dir,
    tokenizer,
    backend_constraint,
    edge_model,
    plan,
):
    """Write TFLite files, bundle the tokenizer, and assemble ``.litertlm``."""
    prefill_tflite_path = os.path.join(temp_dir, "model.tflite")
    edge_model.export(prefill_tflite_path)

    tokenizer_path = _materialize_sentencepiece_tokenizer(tokenizer, temp_dir)

    meta_path = os.path.join(temp_dir, "llm_metadata.pb")
    _build_llm_metadata(
        plan.spec,
        tokenizer,
        plan.cache_length,
        meta_path,
        sampler_config=plan.sampler_config,
        model_type_overridden=plan.model_type_overridden,
    )

    litert_lm_builder = _import_litert_lm_builder()
    builder = litert_lm_builder.LitertLmFileBuilder()
    builder.add_system_metadata(
        litert_lm_builder.Metadata(
            key="Authors",
            value="KerasHub",
            dtype=litert_lm_builder.DType.STRING,
        )
    )
    builder.add_tflite_model(
        prefill_tflite_path,
        litert_lm_builder.TfLiteModelType.PREFILL_DECODE,
        backend_constraint=backend_constraint,
    )
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    builder.add_llm_metadata(meta_path)

    # Write to a temp file in the same directory as `path` and atomically
    # rename it into place on success, so a crash mid-build (the bundle can be
    # large) never leaves a truncated `.litertlm` file at the destination.
    output_dir = os.path.dirname(os.path.abspath(path)) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    try:
        # `mkstemp` always creates the file 0600. Match the permissions a
        # plain `open(path, "wb")` would have produced (0666 minus umask) so
        # switching to an atomic write doesn't silently make bundles
        # unreadable by other users/services that consumed them before.
        umask = os.umask(0)
        os.umask(umask)
        os.fchmod(tmp_fd, 0o666 & ~umask)
        with os.fdopen(tmp_fd, "wb") as output_file:
            builder.build(output_file)
            plan=plan,
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
