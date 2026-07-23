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

# Quantization recipes and attributes are long, stable reference material.
# Keeping them at module level keeps the ``export_to_litertlm`` docstring
# focused on the API while still making the details greppable.
_QUANTIZATION_RECIPES_NOTE = """
Supported ``quant_config`` recipes (from
``litert_torch.generative.quantize.quant_recipes``):

- ``full_dynamic_recipe()`` — dynamic-range quantization of weights
  (activations stay FP32). Recommended default.
- ``full_weight_only_recipe()`` — weight-only quantization. Weights are
  statically quantized; activations remain FP32.
- ``full_fp16_recipe()`` — FP16 weights and activations.

Each recipe accepts the following parameters:

- ``mcfg`` — optional ``ModelConfig`` for the target model. Usually omitted
  for KerasHub exports.
- ``weight_dtype`` — one of:
  ``quant_attrs.Dtype.INT8`` (default),
  ``quant_attrs.Dtype.INT4``,
  ``quant_attrs.Dtype.FP16``,
  ``quant_attrs.Dtype.FP32``.
- ``granularity`` — one of:
  ``quant_attrs.Granularity.CHANNELWISE`` (default),
  ``quant_attrs.Granularity.BLOCKWISE_32``,
  ``quant_attrs.Granularity.BLOCKWISE_64``,
  ``quant_attrs.Granularity.BLOCKWISE_128``,
  ``quant_attrs.Granularity.BLOCKWISE_256``.

Example configurations:

```python
from litert_torch.generative.quantize.quant_recipes import (
    full_dynamic_recipe,
    full_weight_only_recipe,
)
import litert_torch.generative.quantize.quant_attrs as quant_attrs

# Dynamic INT8 weights, FP32 activations (good balance)
quant_config = full_dynamic_recipe()

# Weight-only INT4 (smallest size)
quant_config = full_weight_only_recipe(
    weight_dtype=quant_attrs.Dtype.INT4
)

# Weight-only INT8 with block-wise granularity
quant_config = full_weight_only_recipe(
    weight_dtype=quant_attrs.Dtype.INT8,
    granularity=quant_attrs.Granularity.BLOCKWISE_128,
)
```
"""

# ``litert_torch`` is an optional dependency. Use ``find_spec`` to check for
# availability without importing it at module level, because importing
# ``litert_torch`` has the side effect of enabling ``jax_enable_x64``.
_LITERT_TORCH_AVAILABLE = importlib.util.find_spec("litert_torch") is not None


@contextlib.contextmanager
def _preserve_jax_x64_state():
    """Preserve the JAX ``jax_enable_x64`` flag around ``litert_torch`` usage.

    ``litert_torch`` internally imports JAX and unconditionally enables
    ``jax_enable_x64``. This breaks dtype-sensitive JAX tests elsewhere in the
    same process. We save the original setting and restore it after conversion.
    """
    try:
        import jax
    except ImportError:
        jax = None
        original_x64 = None
    else:
        original_x64 = jax.config.jax_enable_x64
    try:
        yield
    finally:
        if jax is not None:
            jax.config.update("jax_enable_x64", original_x64)


@contextlib.contextmanager
def _preserve_jax_platforms_state():
    """Preserve the JAX ``jax_platforms`` flag around ``litert_torch`` usage.

    LiteRT-LM's JAX bridge (used internally by ``litert_torch`` during MLIR
    lowering) defaults to the TPU platform if one is visible, but export must
    run on CPU so it does not contend with other processes using the TPU. We
    force ``jax_platforms=cpu`` for the duration of tracing/conversion and
    restore the caller's original setting afterward, mirroring
    ``_preserve_jax_x64_state``.
    """
    try:
        import jax
    except ImportError:
        jax = None
        original_platforms = None
    else:
        original_platforms = jax.config.jax_platforms
    if jax is not None:
        jax.config.update("jax_platforms", "cpu")
    try:
        yield
    finally:
        if jax is not None:
            jax.config.update("jax_platforms", original_platforms)


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


def _validate_quant_config(quant_config):
    """Validate that ``quant_config`` is a litert_torch QuantConfig or None."""
    if quant_config is None:
        return
    try:
        from litert_torch.quantize.quant_config import QuantConfig
    except ImportError:
        return
    if not isinstance(quant_config, QuantConfig):
        raise ValueError(
            "`quant_config` must be an instance of "
            "`litert_torch.quantize.quant_config.QuantConfig` or None. "
            f"Received: {type(quant_config).__name__}."
        )


# A cheap sanity check on `hf_tokenizer_path` compatibility, not exact
# validation: small differences (e.g. a handful of reserved/special tokens)
# are expected and not flagged. Only a difference of more than a few hundred
# tokens *and* a meaningfully different order of magnitude (a >=5x ratio)
# together indicate the bundled tokenizer almost certainly does not match
# the model's embedding table -- e.g. bundling a ~32k-vocab Llama tokenizer
# with a model whose embedding table was sized for a ~256k-vocab Gemma
# tokenizer.
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

    Returns a ``(prefill_seq_lens, backend_constraint)`` tuple: the
    normalized list of prefill sequence lengths, and the normalized
    (lowercased) ``backend_constraint`` string (or ``None``). Callers must
    use the returned ``backend_constraint`` -- not the original argument --
    so the lowercased value actually reaches ``_assemble_bundle`` /
    ``builder.add_tflite_model``. Importing ``litert_torch`` is deferred to
    the orchestrator so that the JAX ``jax_enable_x64`` side effect can be
    kept under one preserve/restore context that covers both import and
    tracing.
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
        valid_backends = {"cpu", "gpu", "npu", "gpu_artisan"}
        if backend_constraint not in valid_backends:
            raise ValueError(
                f"Invalid backend_constraint: {backend_constraint!r}. "
                f"Must be one of {sorted(valid_backends)}."
            )

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
    elif _is_sentencepiece_tokenizer(tokenizer):
        _validate_sentencepiece_tokenizer(tokenizer)
    elif isinstance(tokenizer, BytePairTokenizer):
        # Any BytePairTokenizer subclass can be converted to HF tokenizer.json.
        pass
    else:
        raise ValueError(
            "LiteRT-LM export supports SentencePiece tokenizers and any "
            "BytePairTokenizer subclass. Received: "
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

    # LiteRT-LM export relies on litert_torch, which only supports the
    # PyTorch Keras backend. Surface the backend error early, but only after
    # tokenizer validation so that ``test_litertlm_export_unsupported`` tests
    # on other backends still receive the tokenizer-specific error they assert.
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
    pipeline (resolving the model-family spec, cache config, and
    vision/audio config), then passes a single ``ExportPlan`` to the later
    pipeline phases (building sample inputs, tracing/converting, assembling
    the bundle) instead of a long, order-sensitive positional-argument list.
    """

    spec: object
    num_layers: int
    cache_length: int
    num_kv_heads: int
    head_dim: int
    cache_layout: str
    prefill_seq_lens: list
    dtype: object
    has_vision: bool
    has_audio: bool
    vision_cfg: dict | None
    audio_cfg: dict | None
    vision_input_style: str | None
    vision_output_dim: int | None
    max_images: int | None
    tokens_per_image: int | None
    separate_vision_encoder: bool
    sampler_config: object | None
    model_type_overridden: bool


def _build_prefill_inputs(plan):
    """Build a ``{seq_len: sample_inputs}`` map for every prefill bucket."""
    prefill_inputs_map = {}
    for seq_len in plan.prefill_seq_lens:
        base = _build_sample_inputs(
            batch_size=1,
            seq_len=seq_len,
            num_layers=plan.num_layers,
            cache_length=plan.cache_length,
            num_kv_heads=plan.num_kv_heads,
            head_dim=plan.head_dim,
            dtype=plan.dtype,
            cache_layout=plan.cache_layout,
        )
        if plan.has_vision:
            vision_cfg = plan.vision_cfg
            max_images = vision_cfg["max_images_per_prompt"]
            num_vision_tokens = vision_cfg["num_vision_tokens"]
            if plan.separate_vision_encoder:
                tokens_per_image = (
                    num_vision_tokens // max_images if max_images else 1
                )
                base.update(
                    {
                        "mm_embedding": torch.zeros(
                            (
                                1 * max_images,
                                tokens_per_image,
                                plan.vision_output_dim,
                            ),
                            dtype=plan.dtype,
                            device="cpu",
                        ),
                        "vision_indices": torch.zeros(
                            (1, num_vision_tokens), dtype=torch.int32
                        ),
                        "vision_mask": torch.zeros(
                            (1, seq_len), dtype=torch.int32
                        ),
                    }
                )
            elif plan.vision_input_style == "patch_values":
                base.update(
                    _build_gemma4_vision_sample_inputs(
                        batch_size=1,
                        max_images=max_images,
                        patch_size=vision_cfg.get("patch_size", 16),
                        image_size=vision_cfg["image_size"],
                        num_vision_tokens=num_vision_tokens,
                        seq_len=seq_len,
                        dtype=plan.dtype,
                    )
                )
            else:
                # "raw_images" (Gemma3/PaliGemma) and "embedded_pixel_values"
                # (Gemma3n) both need the same raw `[B, N, H, W, 3]` images
                # sample tensor here -- they only diverge in how the adapter
                # *runs* the vision encoder at trace time (see
                # `KerasHubLiteRTAdapter._prepare_image_embeddings`), not in
                # what shape the prefill signature needs.
                base.update(
                    _build_vision_sample_inputs(
                        batch_size=1,
                        max_images=max_images,
                        image_size=vision_cfg["image_size"],
                        num_vision_tokens=num_vision_tokens,
                        seq_len=seq_len,
                        dtype=plan.dtype,
                    )
                )
        if plan.has_audio:
            audio_cfg = plan.audio_cfg
            base.update(
                _build_audio_sample_inputs(
                    batch_size=1,
                    max_clips=audio_cfg["max_clips_per_prompt"],
                    num_frames=audio_cfg["num_frames"],
                    num_audio_tokens=audio_cfg["num_audio_tokens"],
                    seq_len=seq_len,
                    audio_input_feat_size=audio_cfg.get(
                        "audio_input_feat_size", 128
                    ),
                    dtype=plan.dtype,
                )
            )
        prefill_inputs_map[seq_len] = base
    return prefill_inputs_map


def _build_vision_encoder_sample_inputs(
    batch_size,
    max_images,
    image_size,
    patch_size,
    dtype,
    vision_input_style="raw_images",
):
    """Create concrete sample inputs for a separate vision-encoder signature."""
    device = "cpu"
    if vision_input_style == "patch_values":
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size**2 * 3
        return {
            "pixel_values": torch.zeros(
                (
                    batch_size,
                    max_images,
                    num_patches,
                    patch_dim,
                ),
                dtype=dtype,
                device=device,
            ),
            "pixel_position_ids": torch.zeros(
                (
                    batch_size,
                    max_images,
                    num_patches,
                    2,
                ),
                dtype=torch.int32,
                device=device,
            ),
        }
    return {
        "images": torch.zeros(
            (
                batch_size,
                max_images,
                image_size,
                image_size,
                3,
            ),
            dtype=dtype,
            device=device,
        )
    }


def _build_vision_adapter_sample_inputs(
    batch_size,
    max_images,
    tokens_per_image,
    vision_output_dim,
    dtype,
):
    """Create concrete sample inputs for a separate vision-adapter signature."""
    return {
        "features": torch.zeros(
            (
                batch_size * max_images,
                tokens_per_image,
                vision_output_dim,
            ),
            dtype=dtype,
            device="cpu",
        )
    }


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
    model,
    tokenizer,
    prefill_adapter,
    decode_adapter,
    prefill_inputs_map,
    decode_inputs,
    plan,
    quant_config,
    **kwargs,
):
    """Trace prefill/decode (and optional vision) signatures and convert."""
    # Defer torch-specific adapter imports until the backend has been verified
    # as torch, so that non-torch callers get the friendly backend error.
    from keras_hub.src.utils.litertlm.adapter import KerasHubEndOfImageAdapter
    from keras_hub.src.utils.litertlm.adapter import KerasHubVisionAdapter
    from keras_hub.src.utils.litertlm.adapter import (
        KerasHubVisionEncoderAdapter,
    )
    from keras_hub.src.utils.litertlm.traceable_ops import traceable_ops_scope

    with traceable_ops_scope():
        vision_encoder_edge = None
        vision_adapter_edge = None
        eoi_edge = None

        # Optionally export the vision encoder and adapter as separate models.
        if plan.separate_vision_encoder and plan.has_vision:
            vision_cfg = plan.vision_cfg
            patch_size = vision_cfg.get("patch_size", 16)
            vision_encoder_inputs = _build_vision_encoder_sample_inputs(
                batch_size=1,
                max_images=plan.max_images,
                image_size=vision_cfg["image_size"],
                patch_size=patch_size,
                dtype=plan.dtype,
                vision_input_style=plan.vision_input_style,
            )
            vision_adapter_inputs = _build_vision_adapter_sample_inputs(
                batch_size=1,
                max_images=plan.max_images,
                tokens_per_image=plan.tokens_per_image,
                vision_output_dim=plan.vision_output_dim,
                dtype=plan.dtype,
            )
            vision_encoder_adapter = KerasHubVisionEncoderAdapter(
                model,
                vision_input_style=plan.vision_input_style,
                flatten_image_batch=plan.spec.flatten_image_batch,
            ).eval()
            vision_adapter = KerasHubVisionAdapter().eval()

            vision_encoder_edge = litert_torch.signature(
                "vision_encoder",
                vision_encoder_adapter,
                sample_kwargs=vision_encoder_inputs,
                **kwargs,
            ).convert(quant_config=quant_config, lightweight_conversion=True)
            vision_adapter_edge = litert_torch.signature(
                "vision_adapter",
                vision_adapter,
                sample_kwargs=vision_adapter_inputs,
                **kwargs,
            ).convert(quant_config=quant_config, lightweight_conversion=True)

            # Optionally export an END_OF_VISION model: a small, input-less
            # signature whose only output is the family's end-of-image
            # token embedding, mirroring litert-torch's `export_hf` module
            # (see `KerasHubEndOfImageAdapter`'s docstring). Only families
            # that declare `LiteRTLMExportSpec.end_of_vision_token` and
            # whose tokenizer actually resolves it opt in; every other
            # family's bundle is unaffected (`eoi_edge` stays `None`, so
            # `_assemble_bundle` below adds no `END_OF_VISION` section).
            eoi_token_ids = plan.spec.get_end_of_vision_token_ids(tokenizer)
            if eoi_token_ids is not None:
                eoi_adapter = KerasHubEndOfImageAdapter(
                    model, eoi_token_ids
                ).eval()
                eoi_edge = litert_torch.signature(
                    "end_of_vision",
                    eoi_adapter,
                    sample_kwargs={},
                    **kwargs,
                ).convert(
                    quant_config=quant_config, lightweight_conversion=True
                )
            elif plan.spec.end_of_vision_token is not None:
                # The family declares an EOI token, but it did not resolve
                # to a real id in this tokenizer's vocabulary (e.g. a test
                # fixture with a truncated vocab). Skip the section rather
                # than bundle an embedding for the wrong (unknown) token --
                # an absent END_OF_VISION section is a no-op; a wrong one
                # would silently ship an incorrect embedding under a
                # section name the runtime is expected to trust.
                warnings.warn(
                    "Could not resolve end-of-image token "
                    f"{plan.spec.end_of_vision_token!r} to a token id for "
                    f"{type(model).__name__}'s tokenizer; skipping the "
                    "END_OF_VISION bundle section.",
                    UserWarning,
                    stacklevel=2,
                )

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
        edge_model = converter.convert(
            quant_config=quant_config, lightweight_conversion=False
        )

    return edge_model, vision_encoder_edge, vision_adapter_edge, eoi_edge


def _assemble_bundle(
    *,
    path,
    temp_dir,
    tokenizer,
    backend_constraint,
    edge_model,
    vision_encoder_edge,
    vision_adapter_edge,
    eoi_edge,
    plan,
    hf_tokenizer_path,
):
    """Write TFLite files, bundle the tokenizer, and assemble ``.litertlm``."""
    eoi_tflite_path = None
    if plan.separate_vision_encoder and plan.has_vision:
        prefill_tflite_path = os.path.join(temp_dir, "prefill_decode.tflite")
        edge_model.export(prefill_tflite_path)
        vision_encoder_tflite_path = os.path.join(
            temp_dir, "vision_encoder.tflite"
        )
        vision_encoder_edge.export(vision_encoder_tflite_path)
        vision_adapter_tflite_path = os.path.join(
            temp_dir, "vision_adapter.tflite"
        )
        vision_adapter_edge.export(vision_adapter_tflite_path)
        if eoi_edge is not None:
            eoi_tflite_path = os.path.join(temp_dir, "end_of_vision.tflite")
            eoi_edge.export(eoi_tflite_path)
    else:
        prefill_tflite_path = os.path.join(temp_dir, "model.tflite")
        edge_model.export(prefill_tflite_path)

    if hf_tokenizer_path is not None:
        tokenizer_path = hf_tokenizer_path
        use_hf_tokenizer = True
    elif _is_sentencepiece_tokenizer(tokenizer):
        tokenizer_path = _materialize_sentencepiece_tokenizer(
            tokenizer, temp_dir
        )
        use_hf_tokenizer = False
    else:
        tokenizer_path = materialize_hf_tokenizer_json(tokenizer, temp_dir)
        use_hf_tokenizer = True

    meta_path = os.path.join(temp_dir, "llm_metadata.pb")
    _build_llm_metadata(
        plan.spec,
        tokenizer,
        plan.cache_length,
        meta_path,
        vision_cfg=plan.vision_cfg,
        audio_cfg=plan.audio_cfg,
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
    if plan.separate_vision_encoder and plan.has_vision:
        builder.add_tflite_model(
            vision_encoder_tflite_path,
            litert_lm_builder.TfLiteModelType.VISION_ENCODER,
            backend_constraint=backend_constraint,
        )
        builder.add_tflite_model(
            vision_adapter_tflite_path,
            litert_lm_builder.TfLiteModelType.VISION_ADAPTER,
            backend_constraint=backend_constraint,
        )
        if eoi_tflite_path is not None:
            # Ordered after VISION_ADAPTER, matching upstream's section
            # order (`export_hf/core/litert_lm_builder.py`:
            # PREFILL_DECODE -> ... -> VISION_ENCODER -> VISION_ADAPTER ->
            # END_OF_VISION). See `KerasHubEndOfImageAdapter`'s docstring
            # for why this section exists and which families emit it.
            builder.add_tflite_model(
                eoi_tflite_path,
                litert_lm_builder.TfLiteModelType.END_OF_VISION,
                backend_constraint=backend_constraint,
            )
    if use_hf_tokenizer:
        builder.add_hf_tokenizer(tokenizer_path)
    else:
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
    except BaseException:
        os.remove(tmp_path)
        raise
    os.replace(tmp_path, path)

    return path


def export_to_litertlm(
    model,
    path,
    backend_constraint=None,
    prefill_seq_len=None,
    cache_length=None,
    quant_config=None,
    separate_vision_encoder=False,
    hf_tokenizer_path=None,
    sampler_config=None,
    llm_model_type=None,
    **kwargs,
):
    """Export a KerasHub CausalLM model to a LiteRT-LM bundle.

    This exports the model with ``prefill`` and ``decode`` signatures
    required by the LiteRT-LM executor, bundles the tokenizer (SentencePiece
    ``.spm`` for SentencePiece models, or a HuggingFace ``tokenizer.json``
    produced by auto-converting any ``BytePairTokenizer`` subclass), and
    writes an ``LlmMetadata`` protobuf into the ``.litertlm`` artifact.

    **Multimodal:** When the model has a ``vision_encoder`` (e.g. Gemma3),
    the vision encoder is baked into the prefill signature so that image
    inputs are processed alongside text tokens. The decode signature
    remains text-only because image KV-caches are already seeded after
    prefill.

    When ``separate_vision_encoder=True`` and the model has a vision
    encoder, the vision processing is split into three TFLite models:
    ``VISION_ENCODER`` (raw images/patches -> features),
    ``VISION_ADAPTER`` (features -> ``mm_embedding``), and
    ``PREFILL_DECODE`` (text + ``mm_embedding`` -> KV caches/logits). This
    matches the upstream LiteRT-LM multimodal runtime contract. Families
    that declare an end-of-image token (``LiteRTLMExportSpec.
    end_of_vision_token``, e.g. Gemma3/Gemma4) additionally get a fourth,
    input-less ``END_OF_VISION`` model whose only output is that token's
    embedding, mirroring upstream ``export_hf``'s optional ``eoi.tflite``
    section -- an upstream-parity addition; families with no declared
    end-of-image token (e.g. PaliGemma) emit no such section.

    **Bucketing:** ``prefill_seq_len`` accepts either a single ``int`` or a
    ``list[int]``. When a list is provided (e.g.
    ``[32, 64, 128, 256, 512, 1024]``), the exporter traces one prefill
    signature per bucket. At runtime the LiteRT-LM executor dispatches to
    the smallest bucket that fits the actual prompt, avoiding wasted
    computation on padding. For vision-capable models (Gemma3, Gemma3n,
    Gemma4, PaliGemma), bucketing is currently disabled family-wide: every
    prefill bucket must equal ``cache_length``. This is a conservative
    default (governed by the ``allows_vision_bucketing`` export spec flag),
    not a confirmed per-family limitation -- it was first observed for
    Gemma3's image attention-mask computation and has not been re-assessed
    for the other vision families.

    **Quantization:** ``quant_config`` is forwarded to
    ``litert_torch.convert()`` for in-graph quantization. It must be an
    instance of ``litert_torch.quantize.quant_config.QuantConfig``. See
    ``_QUANTIZATION_RECIPES_NOTE`` for supported recipes and attributes.

    Args:
        model: A KerasHub ``CausalLM`` instance with an attached preprocessor
            and tokenizer.
        path: str. Path to save the ``.litertlm`` file.
        backend_constraint: Optional LiteRT-LM backend constraint, such as
            ``"cpu"`` or ``"gpu"``. Defaults to ``None``.
        prefill_seq_len: int or list[int]. Sequence length(s) used when
            tracing the prefill signature(s). Each value must not exceed
            ``cache_length``. Defaults to ``cache_length`` itself.
        cache_length: Optional int. The KV-cache length (the model's maximum
            context window) to trace the export with. If ``None``, this is
            inferred from ``backbone.max_sequence_length`` when the backbone
            defines it; most backbones (e.g. Gemma, Llama, Mistral, Qwen) do
            not, in which case the exporter falls back to
            ``preprocessor.sequence_length`` and emits a ``UserWarning``,
            since that value is a tokenization default chosen for training or
            preprocessing and is not necessarily the model's true maximum
            context length. Pass this explicitly to avoid the warning and to
            get a cache length independent of the preprocessor. Defaults to
            ``None``.
        quant_config: Optional
            ``litert_torch.quantize.quant_config.QuantConfig`` for
            in-conversion quantization. Defaults to ``None`` (no
            quantization, FP32).
        separate_vision_encoder: bool. If ``True`` and the model has a vision
            encoder, export the vision encoder and a no-op vision adapter as
            separate ``VISION_ENCODER`` and ``VISION_ADAPTER`` TFLite models,
            and have ``PREFILL_DECODE`` consume pre-computed ``mm_embedding``
            tensors instead of raw images. Defaults to ``False``. Either way
            the exported bundle is a complete multimodal model —
            ``PREFILL_DECODE`` always consumes text tokens; this flag only
            controls whether vision is baked into that trace or factored
            into reusable models, never producing a vision-only export.
        hf_tokenizer_path: Optional str. Path to a HuggingFace
            ``tokenizer.json`` file to bundle instead of the model's native
            tokenizer. Use this for BytePair / HuggingFace tokenizers that
            cannot be materialized as a SentencePiece ``.spm`` file. When
            provided, the native tokenizer validation is skipped. If ``None``,
            SentencePiece tokenizers are bundled as ``.spm`` and any
            ``BytePairTokenizer`` subclass is automatically converted to
            ``tokenizer.json``. Defaults to ``None``.
        sampler_config: Optional
            ``keras_hub.src.utils.litertlm.model_specs.SamplerConfig``
            instance. When given, the bundle's ``LlmMetadata.sampler_params``
            field is populated from it (mirroring litert-torch export_hf's
            conditional sampler semantics). The only named preset keras-hub
            ships is ``GREEDY_SAMPLER_CONFIG`` (``top_k=1``), for forcing
            deterministic greedy generation during host-vs-device
            verification. Defaults to ``None``, which leaves
            ``sampler_params`` entirely unset so the runtime chooses its own
            sampling policy.
        llm_model_type: Optional str. Explicit ``LlmMetadata.llm_model_type``
            override for presets that are architecturally identical to another
            family and so cannot be auto-detected by class, config, or
            tokenizer -- currently ``"function_gemma"`` (the
            ``function_gemma_instruct_270m`` preset, which loads as a plain
            ``Gemma3CausalLM`` but must export as the ``function_gemma`` model
            type with its function-calling metadata, not as ``gemma3``).
            Mirrors litert-torch's ``litert_lm_model_type_override``. Defaults
            to ``None`` (auto-detect the family by class).
        **kwargs: Additional kwargs forwarded to ``litert_torch`` signature
            tracing.

    Returns:
        The output ``path``.

    Raises:
        ValueError: If the backend is not ``"torch"``, if ``path`` does not
            end with ``.litertlm``, if the model lacks ``call_with_cache``,
            if ``backend_constraint`` is invalid, if any
            ``prefill_seq_len`` exceeds ``cache_length``, if a multimodal
            model is exported with mismatched ``prefill_seq_len`` values, if
            ``sampler_config`` is not a ``SamplerConfig`` instance, if
            ``llm_model_type`` is not a recognized override, or if the model
            is a non-exportable MTP draft model (``Gemma4AssistantCausalLM``).
        ImportError: If ``litert-torch`` or ``litert-lm-builder`` are not
            installed.
    """
    path = os.fspath(path)
    # Fail fast on MTP draft models, which are not standalone-exportable.
    # `Gemma4AssistantCausalLM` is a multi-token-prediction (MTP) draft model
    # for speculative decoding: its `call_with_cache()` requires a target
    # model's hidden state, last-token embedding, and borrowed KV cache, so it
    # has no self-contained prefill/decode graph to export. It subclasses
    # `CausalLM` directly (not `Gemma4CausalLM`), so `resolve_export_spec`
    # would otherwise silently fall through to the generic spec and crash deep
    # in tracing. Reject it here, before tokenizer/backend checks, with a
    # clear message. See `Gemma4AssistantCausalLM` docstring ("This model must
    # NOT be used standalone").
    from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
        Gemma4AssistantCausalLM,
    )

    if isinstance(model, Gemma4AssistantCausalLM):
        raise ValueError(
            f"LiteRT-LM export does not support `{type(model).__name__}`: it "
            "is a multi-token-prediction (MTP) draft model for speculative "
            "decoding, not a standalone model. Its `call_with_cache()` "
            "depends on a target model's hidden state, last-token embedding, "
            "and borrowed KV cache, so it cannot be exported on its own. "
            "Export the target `Gemma4CausalLM` instead; the runtime uses "
            "the draft model via `target_model.generate(..., "
            "assistant_model=...)`."
        )
    if sampler_config is not None and not isinstance(
        sampler_config, SamplerConfig
    ):
        raise ValueError(
            "`sampler_config` must be a "
            "`keras_hub.src.utils.litertlm.model_specs.SamplerConfig` "
            "instance (e.g. `GREEDY_SAMPLER_CONFIG`), got "
            f"{type(sampler_config).__name__}."
        )
    tokenizer = _get_tokenizer(model)
    # `_validate_export_args` returns the normalized (lowercased)
    # `backend_constraint` alongside `prefill_seq_lens`; rebind the local
    # here so the normalized value -- not the original, possibly
    # mixed-case, argument -- is what flows into `_assemble_bundle` /
    # `builder.add_tflite_model` below.
    prefill_seq_lens, backend_constraint = _validate_export_args(
        model,
        path,
        tokenizer,
        backend_constraint,
        hf_tokenizer_path,
        prefill_seq_len,
    )

    # Defer torch-specific adapter imports until after the backend check so
    # that a JAX/TF caller without torch gets the friendly backend error.
    from keras_hub.src.utils.litertlm.adapter import KerasHubLiteRTAdapter
    from keras_hub.src.utils.litertlm.adapter import _cpu_default_device_scope
    from keras_hub.src.utils.litertlm.adapter import _get_vision_encoder

    # Resolve the model-family export spec once and thread it through the
    # rest of the pipeline (and into the adapter), instead of re-deriving
    # family checks at each site. `llm_model_type`, when given, is an explicit
    # caller override for presets indistinguishable from another family by
    # class (e.g. `function_gemma`, a plain `Gemma3CausalLM`).
    spec = resolve_export_spec(model, llm_model_type=llm_model_type)

    # Fail fast on model families whose cache structure the adapter cannot
    # build. Every currently-supported family uses a single stacked KV-cache
    # tensor ("single_stacked"); see `LiteRTLMExportSpec.cache_structure` and
    # `describe_unsupported_cache_structure` for what a family-specific
    # mismatch (e.g. Qwen3.5's hybrid cache) looks like and how to explain
    # it -- this generic check/message applies to any such family, not just
    # Qwen3.5, so a family-specific explanation belongs on that family's own
    # spec class, not as a growing set of branches here. Checking this here,
    # right after the spec is resolved and before any cache-config
    # derivation or tracing, turns what used to be a cryptic `IndexError`
    # deep inside `LiteRTLMExportSpec.stack_kv_cache` into a clear,
    # documented error raised before any expensive work happens.
    if spec.cache_structure != "single_stacked":
        raise ValueError(
            f"LiteRT-LM export does not support `{type(model).__name__}`: "
            f"`{type(model.backbone).__name__}` "
            f"{spec.describe_unsupported_cache_structure()}"
        )

    cache_cfg = spec.get_cache_config(model, cache_length=cache_length)
    num_layers = cache_cfg["num_layers"]
    cache_length = cache_cfg["cache_length"]
    num_kv_heads = cache_cfg["num_kv_heads"]
    head_dim = cache_cfg["head_dim"]
    cache_layout = cache_cfg["cache_layout"]
    if cache_cfg["used_preprocessor_fallback"]:
        warnings.warn(
            "`cache_length` was not specified and "
            f"`{type(model.backbone).__name__}` does not define "
            "`max_sequence_length`. Falling back to "
            f"`preprocessor.sequence_length` ({cache_length}) as the "
            "KV-cache length. This is a tokenization default, not "
            "necessarily the model's true maximum context length. Pass "
            "`cache_length` explicitly to `export_to_litertlm` / "
            '`model.export(..., format="litertlm")` to set it directly.',
            stacklevel=2,
        )

    # Prefill seq_len values must be validated against the real cache length.
    if prefill_seq_lens is None:
        prefill_seq_lens = [cache_length]
    for seq_len in prefill_seq_lens:
        if seq_len > cache_length:
            raise ValueError(
                f"prefill_seq_len ({seq_len}) cannot exceed "
                f"cache_length ({cache_length})."
            )

    # Detect multimodal capabilities.
    vision_cfg = spec.get_vision_config(model)
    audio_cfg = spec.get_audio_config(model)
    has_vision = vision_cfg is not None
    has_audio = audio_cfg is not None

    vision_input_style = None
    vision_output_dim = None
    if has_vision:
        vision_encoder = _get_vision_encoder(model.backbone)
        vision_input_style = spec.vision_input_style
        vision_output_dim = spec.get_vision_output_dim(vision_encoder)
        if separate_vision_encoder and vision_output_dim is None:
            raise ValueError(
                "LiteRT-LM separate vision encoder export requires "
                "`vision_encoder.output_dim` or `vision_encoder.num_classes`."
            )
    elif separate_vision_encoder:
        raise ValueError(
            "`separate_vision_encoder=True` requires a model with a vision "
            "encoder."
        )

    # Some vision families run their encoder *inside* the backbone
    # (`vision_input_style == "embedded_pixel_values"`, e.g. Gemma3n), so
    # there is no separable vision encoder to pack as its own bundle
    # section. Whether a family supports the separate-vision export path is
    # declared once, per family, on the spec (`supports_separate_vision`) --
    # ask it directly instead of re-deriving the fact from the input style
    # here, so the {baked, separate} support matrix is readable from the
    # spec table rather than reconstructed from this branch.
    if (
        separate_vision_encoder
        and has_vision
        and not spec.supports_separate_vision
    ):
        raise ValueError(
            "`separate_vision_encoder=True` is not supported for "
            f"`{type(model).__name__}`: its vision encoder runs inside the "
            "backbone (it expects raw `pixel_values`, e.g. Gemma3n), so "
            "there is no standalone vision encoder to export as a separate "
            "bundle section. Export it with `separate_vision_encoder=False` "
            "(the default)."
        )

    # Vision-capable models currently require every prefill bucket to equal
    # cache_length. This is enforced family-wide (Gemma3, Gemma3n, Gemma4,
    # PaliGemma) as a conservative default via the `allows_vision_bucketing`
    # spec flag, not a Gemma3-specific rule: the constraint was first observed
    # for Gemma3's image attention-mask computation, but no per-family
    # assessment has confirmed whether the other three vision families
    # actually need cache_length == input_length. Relaxing it for a family
    # that provably does not is a future, numerics-gated change -- flip that
    # family's `allows_vision_bucketing` to True (see the spec field), no edit
    # here required.
    if (
        has_vision
        and not spec.allows_vision_bucketing
        and any(seq_len != cache_length for seq_len in prefill_seq_lens)
    ):
        raise ValueError(
            f"Multimodal LiteRT-LM export currently requires all "
            f"`prefill_seq_len` values ({prefill_seq_lens}) to match the "
            f"cache_length ({cache_length}). This restriction is enforced for "
            f"all vision-capable families (Gemma3, Gemma3n, Gemma4, "
            f"PaliGemma) pending a per-family assessment; it was originally "
            f"characterized as specific to Gemma3's image attention-mask "
            f"computation but has not been verified to be Gemma3-specific. "
            f"Pass a single `prefill_seq_len` equal to `cache_length`."
        )

    # Hoist vision shape values that are used both when building prefill inputs
    # and when exporting a separate vision encoder/adapter. Keeping them outside
    # the loop prevents accidental scope leakage and makes the loop body easier
    # to read.
    max_images = None
    tokens_per_image = None
    if has_vision:
        max_images = vision_cfg["max_images_per_prompt"]
        num_vision_tokens = vision_cfg["num_vision_tokens"]
        tokens_per_image = num_vision_tokens // max_images if max_images else 1

    dtype = _torch_dtype_from_model(model)

    # Phases 1-2 (above) resolve the model-family spec and compute every
    # per-export-run setting. Bundle them into a single immutable plan so the
    # remaining phases (building sample inputs, tracing/converting, and
    # assembling the bundle) take one object instead of a long,
    # order-sensitive positional-argument list.
    plan = ExportPlan(
        spec=spec,
        num_layers=num_layers,
        cache_length=cache_length,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_layout=cache_layout,
        prefill_seq_lens=prefill_seq_lens,
        dtype=dtype,
        has_vision=has_vision,
        has_audio=has_audio,
        vision_cfg=vision_cfg,
        audio_cfg=audio_cfg,
        vision_input_style=vision_input_style,
        vision_output_dim=vision_output_dim,
        max_images=max_images,
        tokens_per_image=tokens_per_image,
        separate_vision_encoder=separate_vision_encoder,
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
            cache_layout=plan.cache_layout,
        )

        adapter = KerasHubLiteRTAdapter(
            model,
            plan.num_layers,
            plan.cache_length,
            separate_vision_encoder=(
                plan.separate_vision_encoder and plan.has_vision
            ),
            export_spec=spec,
        )
        adapter.eval()

        prefill_adapter = _PrefillAdapter(adapter).eval()
        decode_adapter = _DecodeAdapter(adapter).eval()

        with _preserve_jax_x64_state(), _preserve_jax_platforms_state():
            import litert_torch

            _validate_quant_config(quant_config)
            edge_model, vision_encoder_edge, vision_adapter_edge, eoi_edge = (
                _trace_and_convert(
                    litert_torch,
                    model,
                    tokenizer,
                    prefill_adapter,
                    decode_adapter,
                    prefill_inputs_map,
                    decode_inputs,
                    plan,
                    quant_config,
                    **kwargs,
                )
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        _assemble_bundle(
            path=path,
            temp_dir=temp_dir,
            tokenizer=tokenizer,
            backend_constraint=backend_constraint,
            edge_model=edge_model,
            vision_encoder_edge=vision_encoder_edge,
            vision_adapter_edge=vision_adapter_edge,
            eoi_edge=eoi_edge,
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
    dtype=None,
    cache_layout="standard",
):
    """Create concrete sample tensors for one signature.

    ``cache_layout`` controls the per-layer KV-cache shape:

    - ``"standard"``: ``[batch_size, cache_length, num_kv_heads, head_dim]``
    - ``"gemma3n"``: ``[batch_size, num_kv_heads, cache_length, head_dim]``
    """
    if dtype is None:
        dtype = torch.float32
    device = "cpu"
    tokens = torch.zeros(
        (batch_size, seq_len), dtype=torch.int32, device=device
    )
    input_pos = torch.arange(seq_len, dtype=torch.int32, device=device)
    if seq_len == 1:
        input_pos = torch.zeros((1,), dtype=torch.int32, device=device)
    kv_cache = {}
    if cache_layout == "gemma3n":
        shape = (batch_size, num_kv_heads, cache_length, head_dim)
    else:
        shape = (batch_size, cache_length, num_kv_heads, head_dim)
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


def _build_vision_sample_inputs(
    batch_size,
    max_images,
    image_size,
    num_vision_tokens,
    seq_len,
    dtype=None,
):
    """Create concrete vision sample tensors for a prefill signature."""
    if dtype is None:
        dtype = torch.float32
    device = "cpu"
    images = torch.zeros(
        (batch_size, max_images, image_size, image_size, 3),
        dtype=dtype,
        device=device,
    )
    vision_indices = torch.zeros(
        (batch_size, num_vision_tokens), dtype=torch.int32, device=device
    )
    vision_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.int32, device=device
    )
    return {
        "images": images,
        "vision_indices": vision_indices,
        "vision_mask": vision_mask,
    }


def _build_gemma4_vision_sample_inputs(
    batch_size,
    max_images,
    patch_size,
    image_size,
    num_vision_tokens,
    seq_len,
    dtype=None,
):
    """Create concrete Gemma4 vision sample tensors for a prefill signature.

    Gemma4's vision encoder expects pre-processed patches
    (``pixel_values`` + ``pixel_position_ids``) rather than raw RGB images.
    """
    if dtype is None:
        dtype = torch.float32
    device = "cpu"
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * 3
    pixel_values = torch.zeros(
        (batch_size, max_images, num_patches, patch_dim),
        dtype=dtype,
        device=device,
    )
    pixel_position_ids = torch.zeros(
        (batch_size, max_images, num_patches, 2),
        dtype=torch.int32,
        device=device,
    )
    vision_indices = torch.zeros(
        (batch_size, num_vision_tokens), dtype=torch.int32, device=device
    )
    vision_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.int32, device=device
    )
    return {
        "pixel_values": pixel_values,
        "pixel_position_ids": pixel_position_ids,
        "vision_indices": vision_indices,
        "vision_mask": vision_mask,
    }


def _build_audio_sample_inputs(
    batch_size,
    max_clips,
    num_frames,
    num_audio_tokens,
    seq_len,
    audio_input_feat_size=128,
    dtype=None,
):
    """Create concrete audio sample tensors for a prefill signature."""
    if dtype is None:
        dtype = torch.float32
    device = "cpu"
    audio_mel = torch.zeros(
        (batch_size, max_clips, num_frames, audio_input_feat_size),
        dtype=dtype,
        device=device,
    )
    audio_mel_mask = torch.zeros(
        (batch_size, max_clips, num_frames), dtype=torch.int32, device=device
    )
    audio_indices = torch.zeros(
        (batch_size, num_audio_tokens), dtype=torch.int32, device=device
    )
    audio_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.int32, device=device
    )
    return {
        "audio_mel": audio_mel,
        "audio_mel_mask": audio_mel_mask,
        "audio_indices": audio_indices,
        "audio_mask": audio_mask,
    }


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


def _validate_sentencepiece_tokenizer(tokenizer):
    file_assets = set(getattr(tokenizer, "file_assets", []) or [])
    if "vocabulary.spm" not in file_assets:
        raise ValueError(
            "LiteRT-LM export currently supports SentencePiece tokenizers "
            "only. Expected tokenizer assets to include `vocabulary.spm`."
        )


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
    vision_cfg=None,
    audio_cfg=None,
    sampler_config=None,
    model_type_overridden=False,
):
    """Serialize an ``LlmMetadata`` protobuf to *path*."""
    # The protobuf lives under an internal-looking subpackage of
    # ``litert-lm-builder``. This is the only way the upstream package exposes
    # the metadata schema, so we import defensively and surface a clear error
    # if the internal layout changes.
    try:
        from litert_lm_builder.litertlm_builder import llm_metadata_pb2
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

    # Some families additionally mark the end of a *chat turn* with a
    # second, distinct token (e.g. Gemma's ``<end_of_turn>``, Llama3's
    # ``<|eot_id|>``) -- see ``LiteRTLMExportSpec.get_chat_stop_token_ids``.
    # This used to be a single hardcoded ``<end_of_turn>`` lookup here,
    # unconditionally applied to every tokenizer regardless of family (it
    # simply no-opped for non-Gemma tokenizers that don't have the token in
    # vocab) -- silently giving no chat-stop-token to e.g. Llama3 or Qwen
    # families that need a *different* second token. Deduplicate against
    # ``end_token_id`` since some families' override intentionally returns
    # the same id (e.g. Qwen3's ``<|im_end|>``, documenting that it is
    # already the primary EOS rather than a coincidence).
    for extra_id in spec.get_chat_stop_token_ids(tokenizer):
        extra_id = int(extra_id)
        if extra_id not in stop_token_ids:
            stop_token_ids.append(extra_id)

    for stop_id in stop_token_ids:
        meta.stop_tokens.add().token_ids.ids.append(stop_id)

    meta.max_num_tokens = int(max_num_tokens)

    getattr(meta.llm_model_type, spec.model_type).SetInParent()

    # Populate vision fields for supported model types.
    if vision_cfg is not None:
        spec.populate_vision_metadata(meta, vision_cfg)

    # Populate audio fields for supported model types.
    if audio_cfg is not None:
        spec.populate_audio_metadata(meta, audio_cfg)

    # Populate function-calling fields for specs that declare them (currently
    # only `FunctionGemmaSpec`, reached via tokenizer auto-detection).
    # A base-class no-op for every other family, so this adds nothing to
    # existing model types -- the same convention as the vision/audio hooks
    # above. Skipped when `llm_model_type` is an explicit caller override:
    # litert-torch skips its model-specific metadata builder whenever
    # `litert_lm_model_type_override` is set
    # (`export_hf/core/litert_lm_builder.py` ~296), so an override yields the
    # bare `llm_model_type` oneof with no model-specific fields.
    if not model_type_overridden:
        spec.populate_function_gemma_metadata(meta)

    # Sampler defaults are written only when the caller explicitly passes a
    # `sampler_config`; otherwise the field is omitted so the runtime picks
    # its own sampling policy. Mirrors litert-torch export_hf's conditional
    # `sampler_params` population (core/litert_lm_builder.py ~189-232),
    # including its GREEDY(top_k==1)/TOP_P/TOP_K type-derivation precedence.
    # keras-hub ships no default sampler (see model_specs.SamplerConfig).
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
        top_p = sampler_config.top_p
        if sampler_config.top_k is not None:
            sp.k = sampler_config.top_k
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
        elif top_p is not None:
            sp.type = SamplerParameters.TOP_P
        else:
            sp.type = SamplerParameters.TOP_P

    with open(path, "wb") as f:
        f.write(meta.SerializeToString())


def _torch_dtype_from_model(model):
    """Return a ``torch.dtype`` matching the model's compute dtype."""
    compute_dtype = getattr(model, "compute_dtype", None)
    if compute_dtype is None:
        compute_dtype = getattr(model.backbone, "compute_dtype", "float32")
    # compute_dtype may be a string, a Keras DTypePolicy, or a torch dtype.
    if hasattr(compute_dtype, "name"):
        compute_dtype = compute_dtype.name
    elif hasattr(compute_dtype, "value"):
        compute_dtype = compute_dtype.value
    elif isinstance(compute_dtype, torch.dtype):
        return compute_dtype
    torch_dtype = getattr(torch, compute_dtype, None)
    if torch_dtype is None:
        raise ValueError(
            f"Unsupported compute_dtype for LiteRT-LM export: "
            f"{compute_dtype!r}. Expected a PyTorch dtype string."
        )
    if torch_dtype is torch.bfloat16:
        warnings.warn(
            "Exporting with `compute_dtype=bfloat16`. BF16 LiteRT-LM export "
            "is untested; numeric parity with the Keras model and runtime "
            "support are not guaranteed. Consider using float32 (optionally "
            "combined with `quant_config` for post-training quantization) "
            "unless you have independently verified BF16 export for this "
            "model.",
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
