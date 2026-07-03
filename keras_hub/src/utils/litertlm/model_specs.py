"""Model-family export specs for LiteRT-LM export.

Before this module existed, model-family-specific knowledge for LiteRT-LM
export was scattered across roughly seven independent sites in
``export.py`` and ``adapter.py``: an isinstance-based ``LlmModelType``
mapping, a backbone class-name ``str.startswith("Gemma3n")`` check for KV
cache layout, getattr-chain fallbacks for cache/vision/audio config, ad-hoc
vision/audio special-token metadata branches, input-tensor-name sniffing to
detect Gemma4-style vision encoders, a Gemma4-specific ``mm_embedding``
reshape, and a Gemma3n-specific attention-mask override. Each of those sites
had to independently know which model family it was looking at.

``LiteRTLMExportSpec`` centralizes that knowledge. A single spec instance is
resolved once per export call (see ``resolve_export_spec``) from a lazy,
``isinstance``-based registry (mirroring the lazy-import pattern already
used elsewhere in this package), and is threaded through the rest of the
export pipeline and the adapter instead of re-deriving family checks at
each site.

``LiteRTLMExportSpec`` itself is also the fallback used for any model not
explicitly registered below (matching today's "unrecognized models map to
generic_model" behavior). This is a pure refactor of detection/config
lookup -- it does not change behavior for any currently-supported model.
"""

def _first_attr(obj, *names, default=None):
    """Return the first non-``None`` attribute from *obj*, or *default*."""
    if obj is None:
        return default
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


class LiteRTLMExportSpec:
    """Default LiteRT-LM export behavior for a model family.

    Also used, unmodified, as the spec for any model family that is not
    explicitly registered in ``_EXPORT_SPEC_REGISTRY`` below -- this mirrors
    today's fallback to ``"generic_model"`` / the generic getattr-chain
    config lookups for unrecognized backbones.

    Subclasses override the class attributes below and/or the methods
    further down to customize behavior for one model family. These are
    plain class attributes (not ``dataclasses.dataclass`` fields):
    subclasses only ever override the *class-level default*, never set a
    per-instance value at construction time, so a plain attribute avoids
    the pitfall of a dataclass-generated ``__init__`` re-baking in the
    parent's field default over a subclass's override.
    """

    #: The ``LlmMetadata.llm_model_type`` oneof name for this family.
    model_type = "generic_model"
    #: Per-layer KV-cache tensor layout. ``"standard"`` is
    #: ``[batch, cache_length, num_kv_heads, head_dim]``; ``"gemma3n"`` is
    #: ``[batch, num_kv_heads, cache_length, head_dim]``.
    cache_layout = "standard"
    #: Whether the vision encoder expects preprocessed patch tensors
    #: (``pixel_values`` + ``pixel_position_ids``) rather than raw images.
    is_gemma4_vision = False

    # -- Cache / vision / audio config -----------------------------------

    def get_cache_config(self, model, cache_length=None):
        """Extract KV-cache dimensions from the model.

        Args:
            model: The KerasHub ``CausalLM`` being exported.
            cache_length: Optional explicit cache length. When ``None``, the
                cache length is inferred from `backbone.max_sequence_length`
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

    def get_vision_config(self, model):
        """Return vision metadata if *model* has a vision encoder, else
        ``None``."""
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return None
        vision_encoder = getattr(backbone, "vision_encoder", None) or getattr(
            backbone, "vit_encoder", None
        )
        if vision_encoder is None:
            return None
        preprocessor = getattr(model, "preprocessor", None)
        max_images = getattr(preprocessor, "max_images_per_prompt", 1)

        image_size = getattr(backbone, "image_size", None)
        if image_size is None:
            # Gemma3n does not set backbone.image_size; read from the
            # preprocessor image converter first, then fall back to the
            # encoder config.
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
        # Image converters may report a (height, width) tuple; downstream
        # code currently assumes a square image, so use the height as the
        # size.
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0]

        num_vision_tokens_per_image = getattr(
            backbone, "num_vision_tokens_per_image", None
        )
        if num_vision_tokens_per_image is None:
            # PaliGemma exposes the per-image token count via
            # ``image_sequence_length`` rather than
            # ``num_vision_tokens_per_image``.
            num_vision_tokens_per_image = getattr(
                backbone, "image_sequence_length", None
            )
        if num_vision_tokens_per_image is None and preprocessor is not None:
            # Gemma3/Gemma3n expose the per-image token count on the
            # preprocessor.
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
            max_clips = getattr(preprocessor, "max_audios_per_prompt", 1)
        num_frames = getattr(preprocessor, "max_audio_frames", 100)
        num_audio_tokens_per_clip = getattr(
            backbone, "num_audio_tokens_per_clip", None
        )
        if num_audio_tokens_per_clip is None and preprocessor is not None:
            # Gemma3n names this attribute ``num_audio_tokens_per_audio``.
            num_audio_tokens_per_clip = getattr(
                preprocessor, "num_audio_tokens_per_audio", 0
            )
        num_audio_tokens = num_audio_tokens_per_clip * max_clips
        audio_input_feat_size = getattr(
            preprocessor, "audio_input_feat_size", None
        )
        if audio_input_feat_size is None and preprocessor is not None:
            audio_converter = getattr(preprocessor, "audio_converter", None)
            if audio_converter is not None:
                audio_input_feat_size = getattr(
                    audio_converter, "feature_size", 128
                )
        if audio_input_feat_size is None:
            audio_input_feat_size = 128
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
            # PaliGemma's ViT uses ``num_classes`` as the projected vision
            # dimension instead of ``output_dim``.
            dim = getattr(vision_encoder, "num_classes", None)
        return dim

    # -- LlmMetadata population -------------------------------------------

    def populate_vision_metadata(self, meta, vision_cfg):
        """Populate vision-related fields in the ``LlmMetadata`` protobuf.

        Default: no-op. Text-only families and families without a dedicated
        ``LlmModelType`` vision subtype (e.g. generic_model, qwen3,
        pali_gemma) do not get vision metadata populated, matching today's
        behavior.
        """
        del meta, vision_cfg

    def populate_audio_metadata(self, meta, audio_cfg):
        """Populate audio-related fields in the ``LlmMetadata`` protobuf.

        Default: no-op (see ``populate_vision_metadata``).
        """
        del meta, audio_cfg

    # -- Adapter-level multimodal handling ---------------------------------

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


class Gemma3Spec(LiteRTLMExportSpec):
    model_type = "gemma3"

    def populate_vision_metadata(self, meta, vision_cfg):
        _populate_gemma3_family_vision_metadata(
            meta, self.model_type, vision_cfg
        )


class Gemma3nSpec(LiteRTLMExportSpec):
    model_type = "gemma3n"
    cache_layout = "gemma3n"

    def populate_vision_metadata(self, meta, vision_cfg):
        _populate_gemma3_family_vision_metadata(
            meta, self.model_type, vision_cfg
        )

    def populate_audio_metadata(self, meta, audio_cfg):
        del audio_cfg
        subtype = meta.llm_model_type.gemma3n
        subtype.start_of_audio_token.token_str = _AUDIO_START_TOKEN
        subtype.end_of_audio_token.token_str = _AUDIO_END_TOKEN

    def get_forced_call_with_cache_kwargs(self, tokens, cache_length):
        # Gemma3n's attention mask computation requires the padding mask to
        # span the full cache length, otherwise a seq_len shorter than
        # cache_length causes a broadcasting error between the causal and
        # padding masks. During export we always pass full-length valid
        # tokens, so a ones mask of cache length is correct.
        import torch

        return {
            "padding_mask": torch.ones(
                (tokens.shape[0], cache_length),
                dtype=torch.bool,
                device=tokens.device,
            )
        }


class Gemma4Spec(LiteRTLMExportSpec):
    model_type = "gemma4"
    is_gemma4_vision = True

    def populate_vision_metadata(self, meta, vision_cfg):
        image_size = vision_cfg["image_size"]
        patch_size = vision_cfg.get("patch_size")
        pool_size = vision_cfg.get("pool_size")
        subtype = meta.llm_model_type.gemma4
        subtype.start_of_image_token.token_str = _GEMMA4_START_OF_IMAGE_TOKEN
        subtype.end_of_image_token.token_str = _GEMMA4_END_OF_IMAGE_TOKEN
        if patch_size is not None:
            subtype.patch_width = patch_size
            subtype.patch_height = patch_size
            subtype.max_num_patches = (image_size // patch_size) ** 2
        if pool_size is not None:
            subtype.pooling_kernel_size = pool_size

    def populate_audio_metadata(self, meta, audio_cfg):
        del audio_cfg
        subtype = meta.llm_model_type.gemma4
        subtype.start_of_audio_token.token_str = _AUDIO_START_TOKEN
        subtype.end_of_audio_token.token_str = _AUDIO_END_TOKEN

    def reshape_separate_vision_embeddings(
        self, img_embeddings, tokens, preprocessor
    ):
        if img_embeddings is None:
            return None
        # Gemma4 interleaves image embeddings with shape
        # (batch, num_images, tokens_per_image, hidden_dim). The separate
        # vision encoder/adapter produces a flat (batch*num_images, ...)
        # tensor, so reshape it back before passing to the language model.
        max_images = getattr(preprocessor, "max_images_per_prompt", 1)
        batch_size = tokens.shape[0]
        return img_embeddings.reshape(
            batch_size,
            max_images,
            img_embeddings.shape[1],
            img_embeddings.shape[2],
        )


class PaliGemmaSpec(LiteRTLMExportSpec):
    """PaliGemma has no dedicated ``LlmModelType`` vision subtype today.

    Registered explicitly (rather than relying on the ``LiteRTLMExportSpec``
    fallback) purely for discoverability, so its lack of special-cased
    behavior is a documented decision rather than an omission.
    """


class Qwen3FamilySpec(LiteRTLMExportSpec):
    """Qwen3, Qwen3-MoE, and Qwen3.5 all map to the "qwen3" oneof."""

    model_type = "qwen3"


class Qwen2p5FamilySpec(LiteRTLMExportSpec):
    """Qwen and Qwen-MoE (pre-Qwen3 architecture) map to "qwen2p5"."""

    model_type = "qwen2p5"


# Special token strings used when populating vision/audio metadata. Keeping
# them as named constants makes it easy to audit which tokens each model
# family expects and avoids scattering literals through the spec classes.
_GEMMA3_START_OF_IMAGE_TOKEN = "<start_of_image>"
_GEMMA3_END_OF_IMAGE_TOKEN = "<end_of_image>"
_GEMMA4_START_OF_IMAGE_TOKEN = "<|image>"
_GEMMA4_END_OF_IMAGE_TOKEN = "<image|>"
_AUDIO_START_TOKEN = "<|audio>"
_AUDIO_END_TOKEN = "<audio|>"


def _populate_gemma3_family_vision_metadata(meta, model_type, vision_cfg):
    """Shared image-token metadata population for gemma3 and gemma3n."""
    image_size = vision_cfg["image_size"]
    subtype = getattr(meta.llm_model_type, model_type)
    subtype.start_of_image_token.token_str = _GEMMA3_START_OF_IMAGE_TOKEN
    subtype.end_of_image_token.token_str = _GEMMA3_END_OF_IMAGE_TOKEN
    subtype.image_tensor_height = image_size
    subtype.image_tensor_width = image_size


# (module_path, class_name, spec_factory). Imported lazily inside
# ``resolve_export_spec`` to avoid heavy top-level dependencies, the same
# pattern used by the module-level model-type mapping this replaces.
_EXPORT_SPEC_REGISTRY = (
    (
        "keras_hub.src.models.gemma4.gemma4_causal_lm",
        "Gemma4CausalLM",
        Gemma4Spec,
    ),
    (
        "keras_hub.src.models.gemma3n.gemma3n_causal_lm",
        "Gemma3nCausalLM",
        Gemma3nSpec,
    ),
    (
        "keras_hub.src.models.gemma3.gemma3_causal_lm",
        "Gemma3CausalLM",
        Gemma3Spec,
    ),
    (
        "keras_hub.src.models.pali_gemma.pali_gemma_causal_lm",
        "PaliGemmaCausalLM",
        PaliGemmaSpec,
    ),
    (
        "keras_hub.src.models.qwen3_moe.qwen3_moe_causal_lm",
        "Qwen3MoeCausalLM",
        Qwen3FamilySpec,
    ),
    (
        "keras_hub.src.models.qwen3.qwen3_causal_lm",
        "Qwen3CausalLM",
        Qwen3FamilySpec,
    ),
    # NOTE: LlmModelType does not have a dedicated "qwen3_5" field. Qwen3.5
    # is architecturally a Qwen3 variant (hybrid attention decoder in the
    # same family), so it maps to the "qwen3" oneof, matching
    # Qwen3MoeCausalLM above.
    (
        "keras_hub.src.models.qwen3_5.qwen3_5_causal_lm",
        "Qwen3_5CausalLM",
        Qwen3FamilySpec,
    ),
    (
        "keras_hub.src.models.qwen_moe.qwen_moe_causal_lm",
        "QwenMoeCausalLM",
        Qwen2p5FamilySpec,
    ),
    (
        "keras_hub.src.models.qwen.qwen_causal_lm",
        "QwenCausalLM",
        Qwen2p5FamilySpec,
    ),
    # NOTE: LlmModelType does not have a dedicated "llama" field; map Llama
    # checkpoints to generic_model so the protobuf oneof stays valid. (This
    # is also the ``LiteRTLMExportSpec`` default, so this entry only exists
    # to make the mapping explicit/greppable.)
    (
        "keras_hub.src.models.llama.llama_causal_lm",
        "LlamaCausalLM",
        LiteRTLMExportSpec,
    ),
)


def resolve_export_spec(model):
    """Return the ``LiteRTLMExportSpec`` for *model*.

    Uses ``isinstance`` checks against ``_EXPORT_SPEC_REGISTRY`` to avoid
    mis-identifying user-defined subclasses, in registration order (the
    first match wins). Unrecognized models get the default
    ``LiteRTLMExportSpec()`` (``model_type="generic_model"``,
    ``cache_layout="standard"``), matching today's fallback behavior.
    """
    for module_path, class_name, spec_factory in _EXPORT_SPEC_REGISTRY:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
        except ImportError:
            continue
        if isinstance(model, cls):
            return spec_factory()

    return LiteRTLMExportSpec()
