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
from keras_hub.src.utils.litertlm.model_specs import _get_vision_encoder
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
            spec=plan.spec,
        )
        if plan.has_vision:
            vision_cfg = plan.vision_cfg
            max_images = vision_cfg["max_images_per_prompt"]
            num_vision_tokens = vision_cfg["num_vision_tokens"]
            if plan.separate_vision_encoder:
                vision_indices, vision_mask = _build_indices_and_mask(
                    1, num_vision_tokens, seq_len
                )
                base.update(
                    {
                        "mm_embedding": torch.zeros(
                            (
                                max_images,
                                plan.tokens_per_image,
                                plan.vision_output_dim,
                            ),
                            dtype=plan.dtype,
                            device="cpu",
                        ),
                        "vision_indices": vision_indices,
                        "vision_mask": vision_mask,
                    }
                )
            elif plan.vision_input_style == "patch_values":
                base.update(
                    _build_gemma4_vision_sample_inputs(
                        batch_size=1,
                        max_images=max_images,
                        patch_size=vision_cfg["patch_size"],
                        image_size=vision_cfg["image_size"],
                        num_vision_tokens=num_vision_tokens,
                        seq_len=seq_len,
                        dtype=plan.dtype,
                    )
                )
            else:
                # "raw_images"/"embedded_pixel_values" both take a raw
                # `[B, N, H, W, 3]` sample tensor here; they only diverge
                # in how the adapter runs the encoder at trace time.
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
                    audio_input_feat_size=audio_cfg["audio_input_feat_size"],
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
    vision_input_style,
):
    """Create concrete sample inputs for a separate vision-encoder signature."""
    device = "cpu"
    if vision_input_style == "patch_values":
        num_patches, patch_dim = _gemma4_patch_dims(image_size, patch_size)
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
    # The LiteRT-LM runtime rejects encoder inputs that are not 3- or 4-D
    # (it feeds one image per call), so the signature is traced with a
    # single-image [B, H, W, 3] input -- no max_images axis. The adapter
    # reintroduces the N=1 axis before calling the KerasHub encoder.
    return {
        "images": torch.zeros(
            (
                batch_size,
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
    tokens_per_image,
    vision_output_dim,
    dtype,
):
    """Create concrete sample inputs for a separate vision-adapter signature.

    The LiteRT-LM runtime chains encoder -> adapter per image, so the adapter
    consumes a single image's features [B, tokens_per_image, dim] -- no
    max_images axis. (Tracing it with batch_size * max_images mismatches the
    single-image encoder output the runtime feeds it at inference time.)
    """
    return {
        "features": torch.zeros(
            (
                batch_size,
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

    if (
        has_vision
        and not spec.allows_vision_bucketing
        and any(seq_len != cache_length for seq_len in prefill_seq_lens)
    ):
        raise ValueError(
            "Multimodal LiteRT-LM export currently requires all "
            f"`prefill_seq_len` values ({prefill_seq_lens}) to match the "
            f"cache_length ({cache_length}). This restriction is enforced "
            "for all vision-capable families (Gemma3, Gemma3n, Gemma4, "
            "PaliGemma) pending a per-family assessment. Pass a single "
            "`prefill_seq_len` equal to `cache_length`."
        )

    # Hoist vision shape values used both in prefill-input building and in
    # separate vision encoder/adapter export.
    max_images = None
    tokens_per_image = None
    if has_vision:
        max_images = vision_cfg["max_images_per_prompt"]
        num_vision_tokens = vision_cfg["num_vision_tokens"]
        tokens_per_image = num_vision_tokens // max_images if max_images else 1

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
            spec=plan.spec,
        )

        adapter = KerasHubLiteRTAdapter(
            model,
            plan.num_layers,
            plan.cache_length,
            export_spec=spec,
            has_audio=plan.has_audio,
            separate_vision_encoder=(
                plan.separate_vision_encoder and plan.has_vision
            ),
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


def _build_indices_and_mask(batch_size, num_tokens, seq_len):
    """Create the zeroed int32 ``(indices, mask)`` sample-tensor pair."""
    indices = torch.zeros(
        (batch_size, num_tokens), dtype=torch.int32, device="cpu"
    )
    mask = torch.zeros((batch_size, seq_len), dtype=torch.int32, device="cpu")
    return indices, mask


def _gemma4_patch_dims(image_size, patch_size):
    """Return ``(num_patches, patch_dim)`` for Gemma4's flattened patches."""
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size**2 * 3
    return num_patches, patch_dim


def _build_vision_sample_inputs(
    batch_size,
    max_images,
    image_size,
    num_vision_tokens,
    seq_len,
    dtype,
):
    """Create concrete vision sample tensors for a prefill signature."""
    device = "cpu"
    images = torch.zeros(
        (batch_size, max_images, image_size, image_size, 3),
        dtype=dtype,
        device=device,
    )
    vision_indices, vision_mask = _build_indices_and_mask(
        batch_size, num_vision_tokens, seq_len
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
    dtype,
):
    """Create concrete Gemma4 vision sample tensors for a prefill signature.

    Gemma4's vision encoder expects pre-processed patches
    (``pixel_values`` + ``pixel_position_ids``) rather than raw RGB images.
    """
    device = "cpu"
    num_patches, patch_dim = _gemma4_patch_dims(image_size, patch_size)
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
    vision_indices, vision_mask = _build_indices_and_mask(
        batch_size, num_vision_tokens, seq_len
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
    dtype,
    audio_input_feat_size,
):
    """Create concrete audio sample tensors for a prefill signature."""
    device = "cpu"
    audio_mel = torch.zeros(
        (batch_size, max_clips, num_frames, audio_input_feat_size),
        dtype=dtype,
        device=device,
    )
    audio_mel_mask = torch.zeros(
        (batch_size, max_clips, num_frames), dtype=torch.int32, device=device
    )
    audio_indices, audio_mask = _build_indices_and_mask(
        batch_size, num_audio_tokens, seq_len
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

    # Populate vision fields for supported model types.
    if vision_cfg is not None:
        spec.populate_vision_metadata(meta, vision_cfg)

    # Populate audio fields for supported model types.
    if audio_cfg is not None:
        spec.populate_audio_metadata(meta, audio_cfg)

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
