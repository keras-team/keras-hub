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
