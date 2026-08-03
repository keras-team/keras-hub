"""Model-family export specs for LiteRT-LM export.

``LiteRTLMExportSpec`` centralizes per-family export knowledge. A single
spec instance is resolved once per export call (see ``resolve_export_spec``)
from a lazy, ``isinstance``-based registry and threaded through the export
pipeline and the adapter. The base class itself is the fallback for any
model family not explicitly registered (``model_type="generic_model"``).
"""

import dataclasses


def _first_attr(obj, *names, default=None):
    """Return the first non-``None`` attribute from *obj*, or *default*."""
    if obj is None:
        return default
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _get_vision_encoder(backbone):
    """Return the vision encoder from a backbone, or ``None``."""
    return _first_attr(backbone, "vision_encoder", "vit_encoder")


def _require_patch_size(patch_size, source):
    """Return *patch_size*, raising an actionable error when it is ``None``.

    Only ``vision_input_style="patch_values"`` families (Gemma4) derive
    exported shapes from ``patch_size``; the other vision families never read
    it, so the requirement is gated on that style rather than imposed on
    every vision encoder.

    Args:
        patch_size: int or None. The patch size read from the model.
        source: str. Where *patch_size* was read from, named in the error.
    """
    if patch_size is None:
        raise ValueError(
            "`patch_size` must be an int for a `patch_values` vision input "
            "style, which sizes the exported `patch_values` signature and the "
            "`max_num_patches` metadata from it. "
            f"Received: patch_size=None (from {source})."
        )
    return patch_size


# Special token strings used when populating vision/audio metadata.
_GEMMA3_START_OF_IMAGE_TOKEN = "<start_of_image>"
_GEMMA3_END_OF_IMAGE_TOKEN = "<end_of_image>"
_GEMMA4_START_OF_IMAGE_TOKEN = "<|image>"
_GEMMA4_END_OF_IMAGE_TOKEN = "<image|>"
_AUDIO_START_TOKEN = "<|audio>"
_AUDIO_END_TOKEN = "<audio|>"

#: Trace-time frame count for the audio sample inputs; no KerasHub
#: preprocessor exposes a frames attribute, so this fixes the exported
#: ``audio_mel`` signature shape.
_DEFAULT_AUDIO_NUM_FRAMES = 100

# Function-calling ("function_gemma") metadata strings, supplied as
# literals because keras-hub's Gemma3 tokenizer has no HuggingFace
# ``special_tokens_map`` carrying them (proto fields shared with Gemma4).
_FUNCTION_GEMMA_CODE_FENCE_START = "```tool_code"
_FUNCTION_GEMMA_CODE_FENCE_END = "```"
_FUNCTION_GEMMA_FUNCTION_RESPONSE_START = "```tool_output"
_FUNCTION_GEMMA_ESCAPE_TOKEN = "<escape>"


@dataclasses.dataclass(frozen=True)
class SamplerConfig:
    """Sampler defaults embedded in a LiteRT-LM bundle's ``LlmMetadata``.

    Mirrors the conditional ``sampler_params`` semantics of litert-torch's
    ``export_hf``: the proto field is only written when a caller explicitly
    requests it. keras-hub ships no default sampler -- omitting
    ``sampler_config`` leaves the field unset, letting the runtime pick its
    own sampling policy.

    Args:
        top_k: Optional int >= 1. ``top_k == 1`` selects deterministic
            greedy generation (encoded as ``TOP_K``).
        top_p: Optional float in (0.0, 1.0].
        temperature: Optional float >= 0.0.
        seed: Optional int RNG seed.
    """

    top_k: int | None = None
    top_p: float | None = None
    temperature: float | None = None
    seed: int | None = None

    def __post_init__(self):
        if (
            self.top_k is None
            and self.top_p is None
            and self.temperature is None
        ):
            raise ValueError(
                "SamplerConfig requires at least one of `top_k`, `top_p`, or "
                "`temperature` to be set; an all-None config would produce "
                "an empty sampler_params. Omit `sampler_config` entirely to "
                "leave the field unset."
            )
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(
                "SamplerConfig.top_k must be >= 1. "
                f"Received: top_k={self.top_k}."
            )
        if self.top_p is not None and not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                "SamplerConfig.top_p must be in (0.0, 1.0]. "
                f"Received: top_p={self.top_p}."
            )
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError(
                "SamplerConfig.temperature must be >= 0.0. "
                f"Received: temperature={self.temperature}."
            )


#: Deterministic greedy sampling (``top_k == 1``). The only named sampler
#: preset keras-hub ships; exercised by the metadata roundtrip test.
GREEDY_SAMPLER_CONFIG = SamplerConfig(top_k=1)


def _single_stacked_support_description():
    """The shared error tail naming the supported cache structure."""
    return (
        "but the LiteRT-LM adapter only supports "
        '`cache_structure="single_stacked"` (a single stacked '
        "`[batch, num_layers, 2, cache_length, num_kv_heads, head_dim]` "
        "KV-cache tensor)."
    )


class LiteRTLMExportSpec:
    """Default LiteRT-LM export behavior for a model family.

    Also used, unmodified, as the spec for any model family not explicitly
    registered in ``_EXPORT_SPEC_REGISTRY`` (the ``"generic_model"``
    fallback). Subclasses customize behavior via the plain class attributes
    and methods below; plain attributes (not dataclass fields) avoid a
    dataclass-generated ``__init__`` re-baking the parent's field default
    over a subclass's override.
    """

    #: The ``LlmMetadata.llm_model_type`` oneof name for this family.
    model_type = "generic_model"
    #: Per-layer KV-cache tensor layout. ``"standard"`` is
    #: ``[batch, cache_length, num_kv_heads, head_dim]``; ``"gemma3n"`` is
    #: ``[batch, num_kv_heads, cache_length, head_dim]``.
    cache_layout = "standard"
    #: Shape of the ``cache`` argument ``call_with_cache`` expects:
    #: ``"single_stacked"`` is one stacked KV tensor (what the default
    #: ``stack_kv_cache`` builds); any other value fails fast in export.
    cache_structure = "single_stacked"
    #: How the vision encoder consumes its input: ``"raw_images"`` (a raw
    #: ``[B, N, H, W, 3]`` tensor; Gemma3, PaliGemma), ``"patch_values"``
    #: (preprocessed patches; Gemma4), or ``"embedded_pixel_values"``
    #: (encoder runs inside the backbone; Gemma3n).
    vision_input_style = "raw_images"
    #: How the audio encoder consumes its input: ``None`` (no audio),
    #: ``"embedded_mel"`` (encoder inside the backbone; Gemma3n), or
    #: ``"standalone_mel"`` (adapter calls ``backbone.audio_encoder``;
    #: Gemma4).
    audio_input_style = None
    #: Whether the vision encoder accepts only a single 4-D image per call,
    #: so the adapter flattens the ``[B, N, H, W, 3]`` stack (PaliGemma).
    flatten_image_batch = False
    #: The end-of-image (EOI) special-token string, or ``None`` if the
    #: family has none. Consulted only on the separate-vision export path,
    #: where it controls whether an ``END_OF_VISION`` section is emitted.
    end_of_vision_token = None
    #: Whether this family's vision path supports multi-bucket prefill.
    #: Default ``False``: the bucketing ban in ``export_to_litertlm`` is a
    #: conservative family-wide default; relaxing it is numerics-gated.
    allows_vision_bucketing = False
    #: Whether this family supports the separate-vision-encoder export path.
    #: ``False`` for families whose encoder runs inside the backbone
    #: (Gemma3n): there is no separable encoder to export.
    supports_separate_vision = True

    def get_cache_config(self, model, cache_length=None):
        """Extract KV-cache dimensions from the model.

        Args:
            model: ``CausalLM``. The KerasHub model being exported.
            cache_length: int or None. Explicit cache length; when ``None``,
                the cache length is inferred from `backbone.max_sequence_length`
                if the backbone defines it, else from
                `preprocessor.sequence_length` -- the caller is responsible
                for warning about this fallback, since the warning needs
                ``warnings.warn``'s call-site stacklevel to point at the
                public API, not at this method.
        """
        backbone = model.backbone
        num_layers = _first_attr(backbone, "num_layers", "num_hidden_layers")
        if num_layers is None:
            raise ValueError(
                "Could not determine number of layers from model backbone. "
                "Expected `backbone.num_layers` or "
                "`backbone.num_hidden_layers`."
            )

        used_preprocessor_fallback = False
        if cache_length is None:
            cache_length = _first_attr(backbone, "max_sequence_length")
            if cache_length is None:
                preprocessor = getattr(model, "preprocessor", None)
                cache_length = _first_attr(preprocessor, "sequence_length")
                used_preprocessor_fallback = cache_length is not None
        if cache_length is None:
            raise ValueError(
                "Could not determine cache length from model backbone or "
                "preprocessor. Please specify `cache_length` or "
                "`prefill_seq_len`, or ensure the model has "
                "`max_sequence_length`."
            )

        num_kv_heads = _first_attr(
            backbone,
            "num_key_value_heads",
            "num_query_heads",
            "num_heads",
            "num_attention_heads",
        )
        if num_kv_heads is None:
            raise ValueError(
                "Could not determine the number of key/value heads from "
                "model backbone. Expected one of "
                "`backbone.num_key_value_heads`, `backbone.num_query_heads`, "
                "`backbone.num_heads`, or `backbone.num_attention_heads`."
            )

        head_dim = _first_attr(backbone, "head_dim")
        if head_dim is None:
            hidden_dim = _first_attr(backbone, "hidden_dim")
            num_qh = _first_attr(
                backbone,
                "num_query_heads",
                "num_heads",
                "num_attention_heads",
            )
            if hidden_dim is not None and num_qh is not None and num_qh > 0:
                head_dim = hidden_dim // num_qh

        if head_dim is None:
            raise ValueError(
                "Could not determine attention head dimension from model "
                "attributes. Expected `backbone.head_dim` or both "
                "`backbone.hidden_dim` and `backbone.num_query_heads`."
            )

        return {
            "num_layers": num_layers,
            "cache_length": cache_length,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "cache_layout": self.cache_layout,
            "used_preprocessor_fallback": used_preprocessor_fallback,
        }

    def get_kv_cache_sample_shape(
        self, batch_size, cache_length, num_kv_heads, head_dim
    ):
        """Return the per-layer KV-cache sample shape for this family.

        Used by ``export.py``'s sample-input builder to size the flat
        ``kv_cache_k_N``/``kv_cache_v_N`` trace inputs. Default: the
        ``"standard"`` ``cache_layout`` shape, ``[batch, cache_length,
        num_kv_heads, head_dim]``.
        """
        return (batch_size, cache_length, num_kv_heads, head_dim)

    def check_exportable(self, model):
        """Raise ``ValueError`` if *model* is not exportable at all.

        Called by ``export_to_litertlm`` immediately after spec resolution,
        before any argument validation or tracing. Default no-op: every
        registered family is exportable. Non-exportable models (see
        ``Gemma4AssistantSpec``) override this to fail fast with a
        family-specific explanation.
        """
        del model

    def describe_unsupported_cache_structure(self):
        """Explain why ``cache_structure`` isn't ``"single_stacked"``.

        Used by ``export_to_litertlm``'s fail-fast check. Default: a
        generic description naming the actual ``cache_structure`` value and
        the shape the adapter does support; families with a more specific
        explanation (e.g. ``Qwen3_5Spec``) override this.
        """
        return (
            f"requires a {self.cache_structure!r} cache structure, "
            f"{_single_stacked_support_description()} "
            "Support for this cache structure is not yet implemented."
        )

    def get_vision_config(self, model):
        """Return vision metadata if *model* has a vision encoder, else
        ``None``."""
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return None
        vision_encoder = _get_vision_encoder(backbone)
        if vision_encoder is None:
            return None
        preprocessor = getattr(model, "preprocessor", None)
        max_images = self.get_max_images_per_prompt(preprocessor)

        image_size = getattr(backbone, "image_size", None)
        if image_size is None:
            # Gemma3n does not set backbone.image_size; read the
            # preprocessor image converter, then the encoder config.
            image_converter = getattr(preprocessor, "image_converter", None)
            if image_converter is not None:
                image_size = getattr(image_converter, "image_size", None)
            if image_size is None:
                vision_encoder_config = getattr(
                    backbone, "vision_encoder_config", {}
                )
                image_shape = vision_encoder_config.get("image_shape")
                if image_shape is not None:
                    image_size = image_shape[0]
        if image_size is None:
            raise ValueError(
                "Could not determine vision image size. Searched "
                "`backbone.image_size`, "
                "`preprocessor.image_converter.image_size`, and "
                "`backbone.vision_encoder_config.image_shape[0]`."
            )
        # Image converters may report a (height, width) tuple; use the
        # height (downstream assumes a square image).
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]

        num_vision_tokens_per_image = getattr(
            backbone, "num_vision_tokens_per_image", None
        )
        if num_vision_tokens_per_image is None:
            # PaliGemma exposes the count as ``image_sequence_length``.
            num_vision_tokens_per_image = getattr(
                backbone, "image_sequence_length", None
            )
        if num_vision_tokens_per_image is None and preprocessor is not None:
            # Gemma3/Gemma3n expose the count on the preprocessor.
            num_vision_tokens_per_image = getattr(
                preprocessor, "num_vision_tokens_per_image", None
            )
        if num_vision_tokens_per_image is None:
            raise ValueError(
                "Could not determine `num_vision_tokens_per_image`. "
                "Searched `backbone.num_vision_tokens_per_image`, "
                "`backbone.image_sequence_length`, and "
                "`preprocessor.num_vision_tokens_per_image`."
            )
        num_vision_tokens = num_vision_tokens_per_image * max_images
        patch_size = getattr(vision_encoder, "patch_size", None)
        if self.vision_input_style == "patch_values":
            patch_size = _require_patch_size(
                patch_size, f"`{type(vision_encoder).__name__}.patch_size`"
            )
        pool_size = getattr(vision_encoder, "pool_size", None)
        return {
            "max_images_per_prompt": max_images,
            "image_size": image_size,
            "num_vision_tokens": num_vision_tokens,
            "num_vision_tokens_per_image": num_vision_tokens_per_image,
            "patch_size": patch_size,
            "pool_size": pool_size,
        }

    def get_audio_config(self, model):
        """Return audio metadata if *model* has an audio encoder, else
        ``None``."""
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return None
        audio_encoder = getattr(backbone, "audio_encoder", None)
        if audio_encoder is None:
            return None
        preprocessor = getattr(model, "preprocessor", None)
        max_clips = getattr(preprocessor, "max_audio_clips_per_prompt", None)
        if max_clips is None:
            # Gemma3n names this attribute ``max_audios_per_prompt``.
            max_clips = getattr(preprocessor, "max_audios_per_prompt", None)
        if max_clips is None:
            raise ValueError(
                "Could not determine `max_clips_per_prompt`. Searched "
                "`preprocessor.max_audio_clips_per_prompt` and "
                "`preprocessor.max_audios_per_prompt`."
            )
        # Trace-time frame count for the audio sample inputs: no KerasHub
        # preprocessor exposes a frames attribute, so this fixes the
        # exported `audio_mel` signature shape.
        num_frames = _DEFAULT_AUDIO_NUM_FRAMES
        num_audio_tokens_per_clip = getattr(
            backbone, "num_audio_tokens_per_clip", None
        )
        if num_audio_tokens_per_clip is None and preprocessor is not None:
            # Gemma3n names this attribute ``num_audio_tokens_per_audio``.
            num_audio_tokens_per_clip = getattr(
                preprocessor, "num_audio_tokens_per_audio", None
            )
        if num_audio_tokens_per_clip is None:
            raise ValueError(
                "Could not determine `num_audio_tokens_per_clip`. Searched "
                "`backbone.num_audio_tokens_per_clip` and "
                "`preprocessor.num_audio_tokens_per_audio`."
            )
        num_audio_tokens = num_audio_tokens_per_clip * max_clips
        audio_input_feat_size = getattr(
            preprocessor, "audio_input_feat_size", None
        )
        if audio_input_feat_size is None and preprocessor is not None:
            audio_converter = getattr(preprocessor, "audio_converter", None)
            if audio_converter is not None:
                audio_input_feat_size = getattr(
                    audio_converter, "feature_size", None
                )
        if audio_input_feat_size is None:
            raise ValueError(
                "Could not determine `audio_input_feat_size`. Searched "
                "`preprocessor.audio_input_feat_size` and "
                "`preprocessor.audio_converter.feature_size`."
            )
        return {
            "max_clips_per_prompt": max_clips,
            "num_frames": num_frames,
            "num_audio_tokens": num_audio_tokens,
            "audio_input_feat_size": audio_input_feat_size,
        }

    def get_vision_output_dim(self, vision_encoder):
        """Return the projected vision feature dimension."""
        dim = getattr(vision_encoder, "output_dim", None)
        if dim is None:
            # PaliGemma's ViT names the projected dimension ``num_classes``.
            dim = getattr(vision_encoder, "num_classes", None)
        return dim

    def get_max_images_per_prompt(self, preprocessor):
        """Return the max images the runtime may pass per prompt.

        Read from ``preprocessor.max_images_per_prompt``. A missing
        attribute is only valid for single-image families
        (``flatten_image_batch=True``, e.g. PaliGemma); on a multi-image
        family it is a misconfiguration and must not silently default to 1.
        """
        max_images = getattr(preprocessor, "max_images_per_prompt", None)
        if max_images is not None:
            return max_images
        if self.flatten_image_batch:
            return 1
        raise ValueError(
            f"{type(self).__name__} declares flatten_image_batch=False "
            "(multi-image family) but its preprocessor has no "
            "`max_images_per_prompt` attribute, so the number of images per "
            "prompt cannot be determined. Set `max_images_per_prompt` on the "
            "preprocessor, or (for a genuinely single-image family) set "
            "`flatten_image_batch = True` on the spec."
        )

    def populate_vision_metadata(self, meta, vision_cfg):
        """Populate vision-related fields in the ``LlmMetadata`` protobuf.

        Default: no-op. Text-only families and families without a dedicated
        ``LlmModelType`` vision subtype populate nothing.
        """
        del meta, vision_cfg

    def populate_audio_metadata(self, meta, audio_cfg):
        """Populate audio-related fields in the ``LlmMetadata`` protobuf.

        Default: no-op (see ``populate_vision_metadata``).
        """
        del meta, audio_cfg

    def populate_function_gemma_metadata(self, meta):
        """Populate function-calling fields in the ``LlmMetadata`` protobuf.

        Default: no-op; only ``FunctionGemmaSpec`` overrides this. Called by
        ``_build_llm_metadata`` except when ``llm_model_type`` was an
        explicit caller override (mirroring litert-torch, which skips its
        model-specific metadata builder on override).
        """
        del meta

    def reshape_separate_vision_embeddings(
        self, img_embeddings, tokens, preprocessor
    ):
        """Reshape ``mm_embedding`` for the separate-vision-encoder path.

        Default: no reshape needed. Only Gemma4 interleaves image
        embeddings with a ``(batch, num_images, tokens_per_image,
        hidden_dim)`` shape that must be restored after the separate
        vision-encoder/adapter models flatten to
        ``(batch * num_images, ...)``.
        """
        del tokens, preprocessor
        return img_embeddings

    def get_forced_call_with_cache_kwargs(self, tokens, cache_length):
        """Return kwargs to force/override on every ``call_with_cache`` call.

        Default: none. Only Gemma3n needs this, to force a full-length
        padding mask (see ``Gemma3nSpec``).
        """
        del tokens, cache_length
        return {}

    # Converting between the flat signature tensors and the family's
    # ``call_with_cache`` cache shape is per-family behavior, so it lives on
    # the spec (a hybrid-cache family overrides these, not the adapter).

    def stack_kv_cache(self, kv_cache, num_layers):
        """Stack flat ``kv_cache_k_N``/``kv_cache_v_N`` tensors into the
        cache format ``call_with_cache`` expects for this family.

        Default: a single ``[batch, num_layers, 2, cache_length,
        num_kv_heads, head_dim]`` tensor. ``torch.stack`` always allocates
        a fresh contiguous tensor, so the result never aliases the input
        buffers and no ``.clone()`` is needed. Torch is imported locally:
        this method is only called after the torch backend is verified.
        """
        import torch

        k_list = [kv_cache[f"kv_cache_k_{i}"] for i in range(num_layers)]
        v_list = [kv_cache[f"kv_cache_v_{i}"] for i in range(num_layers)]
        k_stack = torch.stack(k_list, dim=1)
        v_stack = torch.stack(v_list, dim=1)
        return torch.stack([k_stack, v_stack], dim=2)

    def unstack_kv_cache(self, cache, num_layers):
        """Split the cache ``call_with_cache`` returned back into per-layer
        ``kv_cache_k_N``/``kv_cache_v_N`` output tensors, inverting
        ``stack_kv_cache``.

        Each per-layer tensor is a view into ``cache``, which
        ``call_with_cache`` already returns freshly allocated (Keras's
        cache-update ops are purely functional), and ``torch.export``'s
        functionalization materializes graph-output views into independent
        buffers -- so no ``.clone()`` is needed.
        """
        outputs = {}
        for i in range(num_layers):
            outputs[f"kv_cache_k_{i}"] = cache[:, i, 0, ...]
            outputs[f"kv_cache_v_{i}"] = cache[:, i, 1, ...]
        return outputs

    def get_chat_stop_token_ids(self, tokenizer):
        """Return extra chat-turn-boundary stop token ids for this family.

        ``_build_llm_metadata`` always adds ``tokenizer.end_token_id`` as a
        stop token; families that mark the end of a *chat turn* with a
        distinct second token (Gemma's ``<end_of_turn>``, Llama3's
        ``<|eot_id|>``) override this so on-device chat generation can stop
        at turn boundaries. Default: none. Returned ids need not be
        disjoint from ``end_token_id``; the caller de-duplicates.
        """
        del tokenizer
        return []

    def get_end_of_vision_token_ids(self, tokenizer):
        """Return this family's end-of-image token id(s), or ``None``.

        Used only by the separate-vision-encoder export path to decide
        whether to bundle an ``END_OF_VISION`` model. Returns ``None`` --
        never a wrong, unk-resolved id -- when the family declares no EOI
        token or the tokenizer cannot resolve it, so the caller skips the
        section instead of bundling an incorrect embedding.
        """
        if self.end_of_vision_token is None:
            return None
        token_id = _lookup_token_id(tokenizer, self.end_of_vision_token)
        return [token_id] if token_id is not None else None


def _lookup_token_id(tokenizer, token_str):
    """Return the id for *token_str* in *tokenizer*'s vocab, or ``None``.

    Only looks up when the tokenizer exposes ``token_to_id``, and swallows
    the specific lookup-failure exceptions so a missing special token does
    not abort export. Also treats a lookup that resolves to the tokenizer's
    unk id as "not present", since some tokenizers map unknown lookups to
    the unk id instead of raising.
    """
    if not hasattr(tokenizer, "token_to_id"):
        return None
    try:
        token_id = tokenizer.token_to_id(token_str)
    except (KeyError, ValueError):
        return None
    if token_id is None:
        return None
    unk_id = getattr(tokenizer, "_unk_token_id", None)
    if token_id == unk_id:
        return None
    return token_id


def _gemma_family_chat_stop_token_ids(tokenizer):
    """Return ``[<end_of_turn> id]`` if present in *tokenizer*'s vocab.

    ``<end_of_turn>`` is an optional chat-turn-stop token shared by the
    Gemma family of SentencePiece tokenizers (Gemma, Gemma3, Gemma3n,
    Gemma4, PaliGemma), distinct from ``tokenizer.end_token_id``.
    """
    token_id = _lookup_token_id(tokenizer, "<end_of_turn>")
    return [token_id] if token_id is not None else []


class GemmaSpec(LiteRTLMExportSpec):
    """Base Gemma (Gemma/Gemma2) family.

    There is no dedicated ``LlmModelType`` subtype for base Gemma, so
    ``model_type`` stays ``"generic_model"``. Provides the Gemma-family
    ``<end_of_turn>`` chat-stop-token convention shared by every Gemma*
    spec below (all subclass this instead of ``LiteRTLMExportSpec``).
    """

    def get_chat_stop_token_ids(self, tokenizer):
        return _gemma_family_chat_stop_token_ids(tokenizer)


class Gemma3Spec(GemmaSpec):
    model_type = "gemma3"
    #: Same string as ``LlmModelType.gemma3.end_of_image_token``.
    end_of_vision_token = _GEMMA3_END_OF_IMAGE_TOKEN

    def populate_vision_metadata(self, meta, vision_cfg):
        _populate_gemma3_family_vision_metadata(
            meta, self.model_type, vision_cfg
        )


class FunctionGemmaSpec(Gemma3Spec):
    """The ``function_gemma_instruct_270m`` preset.

    Architecturally identical to Gemma3 -- it loads as a plain
    ``Gemma3CausalLM`` -- so it cannot be distinguished by ``isinstance``
    or config. It is reached via the explicit
    ``llm_model_type="function_gemma"`` override (mirroring litert-torch's
    ``litert_lm_model_type_override``) or by tokenizer auto-detection
    (``_is_function_gemma``). ``model_type = "function_gemma"`` maps it to
    the ``FunctionGemma`` proto instead of ``gemma3``, preserving the
    function-calling metadata a plain Gemma3 export would silently drop.

    Deliberately NOT registered in ``_EXPORT_SPEC_REGISTRY``: an
    ``isinstance`` entry would shadow ``Gemma3Spec`` for *every* Gemma3
    model.
    """

    model_type = "function_gemma"

    def populate_vision_metadata(self, meta, vision_cfg):
        # The ``FunctionGemma`` proto has no image fields, so the gemma3-
        # family vision population inherited from ``Gemma3Spec`` must not run.
        del meta, vision_cfg

    def populate_function_gemma_metadata(self, meta):
        """Populate the ``FunctionGemma`` function-calling proto fields.

        Mirrors litert-torch's Gemma4 metadata builder
        (``export_hf/model_ext/gemma4/metadata_builder.py``), whose
        function-calling field block ``FunctionGemma`` shares (proto fields
        5-14). ``constraint_mode`` is left at its proto default, as
        litert-torch leaves it.
        """
        subtype = meta.llm_model_type.function_gemma
        subtype.code_fence_start = _FUNCTION_GEMMA_CODE_FENCE_START
        subtype.code_fence_end = _FUNCTION_GEMMA_CODE_FENCE_END
        subtype.open_quote = _FUNCTION_GEMMA_ESCAPE_TOKEN
        subtype.close_quote = _FUNCTION_GEMMA_ESCAPE_TOKEN
        subtype.function_response_start = (
            _FUNCTION_GEMMA_FUNCTION_RESPONSE_START
        )
        subtype.use_template_for_fc_format = True


class Gemma3nSpec(GemmaSpec):
    model_type = "gemma3n"
    cache_layout = "gemma3n"
    vision_input_style = "embedded_pixel_values"
    audio_input_style = "embedded_mel"
    #: The encoder runs inside the backbone; no standalone encoder to pack.
    supports_separate_vision = False

    def get_kv_cache_sample_shape(
        self, batch_size, cache_length, num_kv_heads, head_dim
    ):
        """Return Gemma3n's per-layer KV-cache sample shape.

        Gemma3n's ``cache_layout`` transposes the standard shape to
        ``[batch, num_kv_heads, cache_length, head_dim]``.
        """
        return (batch_size, num_kv_heads, cache_length, head_dim)

    def populate_vision_metadata(self, meta, vision_cfg):
        # The gemma3n subtype carries the same image-token fields as gemma3.
        _populate_gemma3_family_vision_metadata(
            meta, self.model_type, vision_cfg
        )

    def populate_audio_metadata(self, meta, audio_cfg):
        del audio_cfg
        # Deliberately Gemma4's audio token strings: that is what the
        # verified golden gemma3n bundles contain. Do NOT "fix" these to
        # Gemma3n's own `<start_of_audio>`/`<end_of_audio>` tokenizer tokens.
        subtype = meta.llm_model_type.gemma3n
        subtype.start_of_audio_token.token_str = _AUDIO_START_TOKEN
        subtype.end_of_audio_token.token_str = _AUDIO_END_TOKEN

    def get_forced_call_with_cache_kwargs(self, tokens, cache_length):
        # Gemma3n's attention-mask computation requires a full-cache-length
        # padding mask (a shorter one mis-broadcasts against the causal
        # mask); export passes full-length valid tokens, so ones is correct.
        import torch

        return {
            "padding_mask": torch.ones(
                (tokens.shape[0], cache_length),
                dtype=torch.bool,
                device=tokens.device,
            )
        }


class Gemma4Spec(GemmaSpec):
    model_type = "gemma4"
    vision_input_style = "patch_values"
    audio_input_style = "standalone_mel"
    #: Same string as ``LlmModelType.gemma4.end_of_image_token``.
    end_of_vision_token = _GEMMA4_END_OF_IMAGE_TOKEN

    def populate_vision_metadata(self, meta, vision_cfg):
        image_size = vision_cfg["image_size"]
        patch_size = _require_patch_size(
            vision_cfg["patch_size"], "`vision_cfg['patch_size']`"
        )
        pool_size = vision_cfg.get("pool_size")
        subtype = meta.llm_model_type.gemma4
        subtype.start_of_image_token.token_str = _GEMMA4_START_OF_IMAGE_TOKEN
        subtype.end_of_image_token.token_str = _GEMMA4_END_OF_IMAGE_TOKEN
        subtype.patch_width = patch_size
        subtype.patch_height = patch_size
        subtype.max_num_patches = (image_size // patch_size) ** 2
        if pool_size is not None:
            subtype.pooling_kernel_size = pool_size

    def populate_audio_metadata(self, meta, audio_cfg):
        # The gemma4 subtype has no proto fields for the derived `audio_cfg`
        # values; they only size the audio trace inputs, so not forwarded.
        del audio_cfg
        subtype = meta.llm_model_type.gemma4
        subtype.start_of_audio_token.token_str = _AUDIO_START_TOKEN
        subtype.end_of_audio_token.token_str = _AUDIO_END_TOKEN
        # `skip_mel_spectrogram_extraction=False` makes the runtime perform
        # mel extraction (`True` feeds raw PCM): the trace consumes log-mel
        # `audio_mel`. Also the proto3 default -- an explicit regression guard.
        subtype.skip_mel_spectrogram_extraction = False

    def reshape_separate_vision_embeddings(
        self, img_embeddings, tokens, preprocessor
    ):
        if img_embeddings is None:
            return None
        # The separate vision encoder/adapter flattens Gemma4's interleaved
        # (batch, num_images, tokens_per_image, hidden_dim) embeddings to
        # (batch*num_images, ...); reshape back before the language model.
        max_images = self.get_max_images_per_prompt(preprocessor)
        batch_size = tokens.shape[0]
        return img_embeddings.reshape(
            batch_size,
            max_images,
            img_embeddings.shape[1],
            img_embeddings.shape[2],
        )


class Gemma4AssistantSpec(LiteRTLMExportSpec):
    """The Gemma4 MTP draft (assistant) model: not standalone-exportable.

    ``Gemma4AssistantCausalLM`` is a multi-token-prediction (MTP) draft
    model for speculative decoding: its ``call_with_cache()`` requires a
    target model's hidden state, last-token embedding, and borrowed KV
    cache, so it has no self-contained prefill/decode graph to export. It
    subclasses ``CausalLM`` directly (not ``Gemma4CausalLM``), so without
    its own registry entry it would fall through to the generic spec and
    crash deep in tracing. See the ``Gemma4AssistantCausalLM`` docstring
    ("This model must NOT be used standalone").
    """

    def check_exportable(self, model):
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


class Llama3Spec(LiteRTLMExportSpec):
    """Llama3's chat template ends a turn with ``<|eot_id|>``.

    ``Llama3Tokenizer`` registers both ``<|end_of_text|>`` (the primary EOS,
    ``end_token_id``) and ``<|eot_id|>`` (as the secondary ``end_token2``)
    because checkpoints have no config indicating the true stop token.
    Without this override, ``<|eot_id|>`` never reaches the exported
    metadata, so on-device chat generation cannot stop at a turn boundary.
    """

    def get_chat_stop_token_ids(self, tokenizer):
        eot_id = getattr(tokenizer, "end_token2_id", None)
        return [eot_id] if eot_id is not None else []


class Phi3Spec(LiteRTLMExportSpec):
    """Phi-3's chat template ends every turn with ``<|end|>``.

    ``<|end|>`` is distinct from the primary EOS ``<|endoftext|>`` (what
    ``Phi3Tokenizer`` registers as ``end_token``), so it never reaches the
    exported metadata without this override. It is an ordinary special
    token in the vocab, so look it up by string and return ``[]`` when it
    is absent (base/non-instruct vocabularies).
    """

    def get_chat_stop_token_ids(self, tokenizer):
        token_id = _lookup_token_id(tokenizer, "<|end|>")
        return [token_id] if token_id is not None else []


# These keep the `LiteRTLMExportSpec` default (EOS-only): Mistral/Mixtral
# end turns with the primary EOS `</s>`; base Llama, GPT2, Bloom, GPT-NeoX
# and OPT are base LMs with no second chat-turn token in their tokenizers.


def _qwen_family_chat_stop_token_ids(tokenizer):
    """Return ``[<|im_end|> id]`` if present in *tokenizer*'s vocab.

    ``<|im_end|>`` is the ChatML chat-turn-stop token shared by the Qwen
    families (Qwen3 and pre-Qwen3 Qwen/Qwen-MoE alike).
    """
    token_id = _lookup_token_id(tokenizer, "<|im_end|>")
    return [token_id] if token_id is not None else []


class Qwen3FamilySpec(LiteRTLMExportSpec):
    """Qwen3, Qwen3-MoE, and Qwen3.5 all map to the "qwen3" oneof."""

    model_type = "qwen3"

    def get_chat_stop_token_ids(self, tokenizer):
        # Qwen3's `<|im_end|>` is already `tokenizer.end_token_id`; surfaced
        # explicitly (`_build_llm_metadata` de-duplicates).
        return _qwen_family_chat_stop_token_ids(tokenizer)


class Qwen3_5Spec(Qwen3FamilySpec):
    """Qwen3.5 spec: maps to "qwen3" but its hybrid cache is unsupported.

    Qwen3.5's hybrid full-attention/linear-attention decoder layers need a
    dual cache (``Qwen3_5CausalLM.call_with_cache`` expects a ``(kv_cache,
    conv_cache, recurrent_cache)`` tuple) that the LiteRT-LM adapter's
    single stacked-KV-tensor cache format cannot represent yet.
    """

    cache_structure = "hybrid"

    def describe_unsupported_cache_structure(self):
        return (
            "requires a 'hybrid' cache structure: Qwen3.5's hybrid "
            "full_attention/linear_attention layers use a dual cache "
            "structure (`call_with_cache` expects a `(kv_cache, conv_cache, "
            "recurrent_cache)` tuple, since linear-attention layers need a "
            "convolutional/recurrent state that a stacked KV tensor cannot "
            f"represent), {_single_stacked_support_description()} "
            "Support for hybrid cache structures is not yet implemented."
        )


class Qwen2p5FamilySpec(LiteRTLMExportSpec):
    """Qwen and Qwen-MoE (pre-Qwen3 architecture) map to "qwen2p5"."""

    model_type = "qwen2p5"

    def get_chat_stop_token_ids(self, tokenizer):
        # Qwen 2.5 registers `<|endoftext|>` as `end_token`, but ChatML
        # checkpoints may still carry `<|im_end|>`; add it when present.
        return _qwen_family_chat_stop_token_ids(tokenizer)


def _populate_gemma3_family_vision_metadata(meta, model_type, vision_cfg):
    """Shared image-token metadata population for gemma3 and gemma3n."""
    image_size = vision_cfg["image_size"]
    subtype = getattr(meta.llm_model_type, model_type)
    subtype.start_of_image_token.token_str = _GEMMA3_START_OF_IMAGE_TOKEN
    subtype.end_of_image_token.token_str = _GEMMA3_END_OF_IMAGE_TOKEN
    subtype.image_tensor_height = image_size
    subtype.image_tensor_width = image_size


# (module_path, class_name, spec_factory), imported lazily inside
# ``resolve_export_spec`` to avoid heavy top-level dependencies.
_EXPORT_SPEC_REGISTRY = (
    ("keras_hub.src.models.gemma4.gemma4_causal_lm", "Gemma4CausalLM", Gemma4Spec),
    ("keras_hub.src.models.gemma3n.gemma3n_causal_lm", "Gemma3nCausalLM", Gemma3nSpec),
    ("keras_hub.src.models.gemma3.gemma3_causal_lm", "Gemma3CausalLM", Gemma3Spec),
    ("keras_hub.src.models.gemma.gemma_causal_lm", "GemmaCausalLM", GemmaSpec),
    ("keras_hub.src.models.qwen3_5.qwen3_5_causal_lm", "Qwen3_5CausalLM", Qwen3_5Spec),
    ("keras_hub.src.models.qwen3.qwen3_causal_lm", "Qwen3CausalLM", Qwen3FamilySpec),
    ("keras_hub.src.models.qwen2_5.qwen2_5_causal_lm", "Qwen2_5CausalLM", Qwen2p5FamilySpec),
    ("keras_hub.src.models.llama3.llama3_causal_lm", "Llama3CausalLM", Llama3Spec),
    ("keras_hub.src.models.phi3.phi3_causal_lm", "Phi3CausalLM", Phi3Spec),
)
_MODEL_TYPE_OVERRIDE_SPECS = {
    "function_gemma": FunctionGemmaSpec,
}


def _is_function_gemma(model):
    """Detect the ``function_gemma_instruct_270m`` preset by tokenizer tokens.

    The preset loads as a plain ``Gemma3CausalLM`` (architecturally identical
    to Gemma3), but its ``Gemma3Tokenizer`` vocabulary contains the
    function-calling special tokens ``<start_function_call>`` and
    ``<end_function_call>``. Plain Gemma3 presets do not.

    We must verify the token round-trips, because some tokenizers map
    out-of-vocabulary strings to a fixed ``<unk>`` id instead of raising.
    """
    try:
        tokenizer = model.preprocessor.tokenizer
        token_id = tokenizer.token_to_id("<start_function_call>")
        return tokenizer.id_to_token(token_id) == "<start_function_call>"
    except (AttributeError, ValueError, KeyError, TypeError):
        return False


def _resolve_export_spec_by_class(model):
    """Return the spec ``_EXPORT_SPEC_REGISTRY`` maps *model*'s class to."""
    # function_gemma loads as a plain Gemma3CausalLM but is distinguished by
    # function-calling special tokens in its tokenizer. Check before the
    # registry so it does not fall through to Gemma3Spec.
    if _is_function_gemma(model):
        return FunctionGemmaSpec()
    for module_path, class_name, spec_factory in _EXPORT_SPEC_REGISTRY:
        # Modules are imported lazily so importing this module stays cheap;
        # all entries are first-party and must import cleanly -- an
        # ImportError here is a real bug and must surface, not be skipped.
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        if isinstance(model, cls):
            return spec_factory()

    return LiteRTLMExportSpec()


def resolve_export_spec(model, llm_model_type=None):
    """Return the ``LiteRTLMExportSpec`` for *model*.

    If *llm_model_type* is given, it is an explicit caller override that
    selects a spec by ``LlmMetadata.llm_model_type`` name for presets that are
    architecturally indistinguishable from another family (e.g.
    ``function_gemma``, which loads as a plain ``Gemma3CausalLM``); see
    ``_MODEL_TYPE_OVERRIDE_SPECS``. Otherwise the spec is resolved by
    ``isinstance`` checks against ``_EXPORT_SPEC_REGISTRY`` (in registration
    order, first match wins, to avoid mis-identifying user-defined
    subclasses). Unrecognized models get the default ``LiteRTLMExportSpec()``
    (``model_type="generic_model"``, ``cache_layout="standard"``).

    Raises:
        ValueError: If *llm_model_type* is not a recognized override, or if it
            is used on a model whose class is not exportable at all (the
            override selects exported metadata, not exportability).
    """
    if llm_model_type is not None:
        try:
            override_spec_factory = _MODEL_TYPE_OVERRIDE_SPECS[llm_model_type]
        except KeyError:
            raise ValueError(
                f"Unknown `llm_model_type` override {llm_model_type!r}. "
                "Supported overrides: "
                f"{sorted(_MODEL_TYPE_OVERRIDE_SPECS)}. Omit the argument to "
                "auto-detect the model family by class."
            )
        # Exportability is a property of the model class, not of the requested
        # metadata subtype, so the gate runs on the class-resolved spec: an
        # override must not smuggle a non-exportable model past it. The other
        # branch returns that same class-resolved spec, so the caller's own
        # `check_exportable` call covers it.
        _resolve_export_spec_by_class(model).check_exportable(model)
        return override_spec_factory()
    return _resolve_export_spec_by_class(model)
