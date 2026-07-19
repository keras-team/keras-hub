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
    #: Shape of the ``cache`` argument ``call_with_cache`` expects.
    #: ``"single_stacked"`` (every currently-supported family) is a single
    #: stacked ``[batch, num_layers, 2, cache_length, num_kv_heads,
    #: head_dim]`` KV-cache tensor -- exactly what the default
    #: ``stack_kv_cache`` below builds. ``"hybrid"`` (Qwen3.5) means
    #: ``call_with_cache`` instead expects a ``(kv_cache, conv_cache,
    #: recurrent_cache)`` tuple, because hybrid full-attention/
    #: linear-attention architectures need two structurally different
    #: per-layer caches. The default ``stack_kv_cache``/``unstack_kv_cache``
    #: do not build/parse that shape (no current spec overrides them to), so
    #: ``export_to_litertlm`` fails fast on any spec with
    #: ``cache_structure != "single_stacked"`` (see the early validation in
    #: ``export.py``) instead of letting a mismatched cache reach
    #: ``call_with_cache`` and fail with a cryptic ``IndexError``.
    cache_structure = "single_stacked"
    #: How this family's vision encoder consumes its input (only meaningful
    #: when `get_vision_config` returns non-``None``). One of:
    #: - ``"raw_images"`` (default): the vision encoder -- or
    #:   ``KerasHubVisionEncoderAdapter`` on the separate-vision-encoder
    #:   export path -- is called with a raw ``[B, N, H, W, 3]`` images
    #:   tensor (Gemma3, PaliGemma).
    #: - ``"patch_values"``: the vision encoder is called with preprocessed
    #:   ``pixel_values`` + ``pixel_position_ids`` patch tensors (Gemma4).
    #: - ``"embedded_pixel_values"``: the vision encoder runs *inside* the
    #:   backbone, so `call_with_cache` itself consumes raw ``pixel_values``
    #:   directly and the adapter never calls a separate vision encoder at
    #:   all (Gemma3n). `separate_vision_encoder=True` is meaningless for
    #:   this style and is rejected in `export_to_litertlm`.
    vision_input_style = "raw_images"
    #: How this family's audio encoder consumes its input (only meaningful
    #: when `get_audio_config` returns non-``None``). One of:
    #: - ``None`` (default): the family has no audio encoder.
    #: - ``"embedded_mel"``: the audio encoder runs *inside* the backbone, so
    #:   `call_with_cache` itself consumes the pre-extracted mel spectrogram
    #:   (``input_features``/``input_features_mask``) directly and the adapter
    #:   never calls a separate audio encoder (Gemma3n).
    #: - ``"standalone_mel"``: the adapter calls
    #:   ``backbone.audio_encoder(audio_mel, audio_mel_mask)`` as a standalone
    #:   in-trace stage and passes the resulting embeddings into
    #:   `call_with_cache` (Gemma4).
    #: This replaces the old ``"input_features" in call_with_cache signature``
    #: sniff in ``adapter.py`` -- the same migration ``vision_input_style``
    #: already made (see the comment near the separate-vision-encoder rejection
    #: in ``export.py``): the spec, resolved once from the family registry,
    #: already knows this about its own family, so ask it instead of
    #: re-deriving the fact from signature introspection at trace time.
    audio_input_style = None

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

    def describe_unsupported_cache_structure(self):
        """Explain why ``cache_structure`` isn't ``"single_stacked"``.

        Used by ``export_to_litertlm``'s fail-fast check (raised right
        after the spec is resolved, before any cache-config derivation or
        tracing) when ``self.cache_structure != "single_stacked"``. Default:
        a generic description naming the actual ``cache_structure`` value
        and the shape the adapter does support. Families with a more
        specific/helpful explanation of *why* their cache doesn't fit (e.g.
        ``Qwen3_5Spec``, whose hybrid attention/linear-attention layers need
        a genuinely different, non-KV-only cache) should override this
        instead of the generic validation in ``export.py`` growing another
        family-specific branch.
        """
        return (
            f"requires a {self.cache_structure!r} cache structure, but the "
            "LiteRT-LM adapter only supports "
            '`cache_structure="single_stacked"` (a single stacked '
            "`[batch, num_layers, 2, cache_length, num_kv_heads, head_dim]` "
            "KV-cache tensor). Support for this cache structure is not yet "
            "implemented."
        )

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

    # -- KV-cache stack/unstack ---------------------------------------------
    #
    # Cache *layout* (the per-layer tensor shape, see ``cache_layout``) and
    # cache *structure* (what ``call_with_cache`` expects as its ``cache``
    # argument overall, see ``cache_structure``) are already per-family spec
    # facts. Stacking/unstacking between the flat ``kv_cache_k_N``/
    # ``kv_cache_v_N`` tensors LiteRT-LM's signature contract uses and
    # whatever shape ``call_with_cache`` actually expects is therefore also
    # family-specific behavior, not a fixed adapter mechanism -- these two
    # methods are what a future ``cache_structure="hybrid"`` family (this is
    # exactly the seam Qwen3.5's hybrid ``(kv_cache, conv_cache,
    # recurrent_cache)`` cache needs -- see ``Qwen3_5Spec``) would override,
    # instead of ``KerasHubLiteRTAdapter`` growing per-family branches.
    #
    # INVESTIGATION (2026-07-03, keras-hub PR #2705 review, item 7 --
    # decode-path KV-cache copy cost): traced a tiny Gemma decode step
    # (num_layers=4, cache_length=16, num_kv_heads=1, head_dim=8; per-layer
    # per-k/v-tensor size P = cache_length * num_kv_heads * head_dim = 128
    # elements) with ``torch.export`` and inspected the graph for
    # ``stack``/``cat`` nodes on cache-sized tensors:
    #
    # - ``stack_kv_cache`` itself produces exactly 3 ``torch.stack`` nodes
    #   per decode step: ``k_stack``/``v_stack`` (``num_layers * P`` = 512
    #   elements each) plus the final ``stack([k_stack, v_stack])`` combine
    #   (``2 * num_layers * P`` = 1024 elements). Total elements written:
    #   ``4 * num_layers * P`` = 2048 -- i.e. **2x** the logical KV-cache
    #   size (``2 * num_layers * P`` = 1024) is copied here per decode
    #   step, purely to convert the flat per-layer ``kv_cache_k_N``/
    #   ``kv_cache_v_N`` signature contract into the single stacked tensor
    #   ``call_with_cache`` expects.
    # - ``unstack_kv_cache`` produced **zero** ``stack``/``cat`` nodes in
    #   the traced graph (only ``select``/``getitem``), confirming it is
    #   genuinely view-based with no extra copy, as its docstring claims.
    # - A further ~2x cache size is copied *inside* ``call_with_cache``
    #   itself (one ``stack([k, v])`` rebuild per layer after that layer's
    #   cache update, plus one final stack across all ``num_layers``
    #   layers' results) -- this is a property of the underlying Keras
    #   model's own per-layer cache-update mechanism, not something these
    #   two methods control, so it is out of scope here.
    #
    # Net: ~4x the logical KV-cache size is copied via ``stack`` somewhere
    # in the traced decode graph per generated token -- 2x attributable to
    # this method, 2x to the underlying model's cache update. For a
    # production-sized cache (many layers x long cache_length x realistic
    # head_dim) this is a genuine, per-token-recurring data-movement cost,
    # not just per-tensor binding overhead. Restructuring it would mean
    # either changing the LiteRT-LM signature contract (the flat
    # ``kv_cache_k_N``/``kv_cache_v_N`` naming convention) or Keras cache
    # internals -- both larger changes than this fix batch's scope, so this
    # is flagged as a follow-up rather than attempted here.

    def stack_kv_cache(self, kv_cache, num_layers):
        """Stack flat ``kv_cache_k_N``/``kv_cache_v_N`` tensors into the
        cache format ``call_with_cache`` expects for this family.

        Default (every family with ``cache_structure="single_stacked"``):
        stack into a single ``[batch, num_layers, 2, cache_length,
        num_kv_heads, head_dim]`` tensor -- exactly what
        ``KerasHubLiteRTAdapter`` used to build inline before this moved
        onto the spec. ``torch.stack`` always allocates a new, contiguous
        tensor -- it never returns a view aliasing its inputs -- so the
        doubly-nested stack below is already guaranteed to be a fresh
        allocation independent of the per-layer ``kv_cache_k_N``/
        ``kv_cache_v_N`` input buffers, without an extra ``.clone()``.

        Torch is imported locally (this method is only ever called from the
        adapter, after the torch backend has already been verified), the
        same pattern ``Gemma3nSpec.get_forced_call_with_cache_kwargs`` uses.
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

        Default: inverse of the single-stacked layout built above. Each
        per-layer tensor is a view into ``cache``, which is itself already
        a fresh, non-aliased tensor produced by ``call_with_cache`` (Keras's
        ``slice_update``/``scatter_update`` cache-update ops are purely
        functional and never mutate their input in place). No additional
        ``.clone()`` is needed here: ``torch.export``'s functionalization
        pass materializes graph-output views into independent buffers
        automatically, exercised by the full litert_lm-runtime generation
        tests (multi-step decode, exactly the scenario where aliased output
        buffers would surface as corrupted generation).
        """
        outputs = {}
        for i in range(num_layers):
            outputs[f"kv_cache_k_{i}"] = cache[:, i, 0, ...]
            outputs[f"kv_cache_v_{i}"] = cache[:, i, 1, ...]
        return outputs

    # -- Metadata: chat-turn stop tokens -----------------------------------

    def get_chat_stop_token_ids(self, tokenizer):
        """Return extra chat-turn-boundary stop token ids for this family.

        ``_build_llm_metadata`` always adds ``tokenizer.end_token_id`` (the
        primary EOS used for packing/training) as a stop token. Some
        families additionally mark the end of a *chat turn* with a second,
        distinct token that is not ``end_token_id`` -- e.g. Gemma's
        ``<end_of_turn>`` (see ``GemmaSpec``) or Llama3's ``<|eot_id|>``
        (see ``Llama3Spec``). Without surfacing that second token, on-device
        chat generation for those families has no way to know when to stop
        at a turn boundary (risk of runaway generation).

        Default: none. This covers every family whose primary EOS *is*
        already the chat-turn-stop token -- e.g. Qwen3's ``<|im_end|>`` is
        already ``tokenizer.end_token_id`` (see ``Qwen3Tokenizer``), so no
        override is needed there for that reason (``Qwen3FamilySpec`` still
        overrides this to make that fact explicit rather than incidental --
        see below).

        Callers must not assume the returned ids are disjoint from
        ``end_token_id``; ``_build_llm_metadata`` de-duplicates before
        writing ``meta.stop_tokens``.
        """
        del tokenizer
        return []


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

    Not registered with its own dedicated ``LlmModelType`` subtype (there is
    no "gemma" oneof field distinct from "gemma3"/"gemma3n"/"gemma4"), so
    ``model_type`` stays the ``LiteRTLMExportSpec`` default of
    ``"generic_model"``, matching today's behavior. Registered explicitly so
    the Gemma-family ``<end_of_turn>`` chat-stop-token convention (shared
    with ``Gemma3Spec``/``Gemma3nSpec``/``Gemma4Spec``/``PaliGemmaSpec``
    below, all of which subclass this instead of ``LiteRTLMExportSpec``
    directly) lives on the registry instead of as an unconditional check in
    ``_build_llm_metadata`` that happened to no-op for non-Gemma tokenizers
    only because they don't have ``<end_of_turn>`` in vocab.
    """

    def get_chat_stop_token_ids(self, tokenizer):
        return _gemma_family_chat_stop_token_ids(tokenizer)


class Gemma3Spec(GemmaSpec):
    model_type = "gemma3"

    def populate_vision_metadata(self, meta, vision_cfg):
        _populate_gemma3_family_vision_metadata(
            meta, self.model_type, vision_cfg
        )


class Gemma3nSpec(GemmaSpec):
    model_type = "gemma3n"
    cache_layout = "gemma3n"
    vision_input_style = "embedded_pixel_values"
    audio_input_style = "embedded_mel"

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


class Gemma4Spec(GemmaSpec):
    model_type = "gemma4"
    vision_input_style = "patch_values"
    audio_input_style = "standalone_mel"

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


class PaliGemmaSpec(GemmaSpec):
    """PaliGemma has no dedicated ``LlmModelType`` vision subtype today.

    Registered explicitly (rather than relying on the ``LiteRTLMExportSpec``
    fallback) purely for discoverability, so its lack of special-cased
    behavior is a documented decision rather than an omission. Subclasses
    ``GemmaSpec`` (not ``LiteRTLMExportSpec`` directly) since PaliGemma uses
    a Gemma tokenizer and shares the same ``<end_of_turn>`` convention.
    """


class Llama3Spec(LiteRTLMExportSpec):
    """Llama3's chat template ends a turn with ``<|eot_id|>``.

    ``Llama3Tokenizer`` stores this as the secondary special token
    ``end_token2`` (see the "Hack" comment in ``llama3_tokenizer.py``):
    Llama3 checkpoints have no config indicating whether the true stop
    token is ``<|end_of_text|>`` or ``<|eot_id|>``, so the tokenizer
    registers both, but the packer always uses ``<|end_of_text|>``
    (``tokenizer.end_token_id``) as the primary EOS. Without this override,
    ``<|eot_id|>`` never reaches the exported metadata, so on-device chat
    generation has no way to know when to stop at a turn boundary.
    """

    def get_chat_stop_token_ids(self, tokenizer):
        eot_id = getattr(tokenizer, "end_token2_id", None)
        return [eot_id] if eot_id is not None else []


class Qwen3FamilySpec(LiteRTLMExportSpec):
    """Qwen3, Qwen3-MoE, and Qwen3.5 all map to the "qwen3" oneof."""

    model_type = "qwen3"

    def get_chat_stop_token_ids(self, tokenizer):
        # Qwen3's chat template ends a turn with `<|im_end|>`, which is
        # already the tokenizer's primary EOS (`tokenizer.end_token_id` --
        # see `Qwen3Tokenizer.__init__`). This override makes that
        # intentional rather than an accident of `end_token_id` happening
        # to already be the right token; `_build_llm_metadata`
        # de-duplicates against `end_token_id` so this does not add a
        # redundant entry to the exported metadata.
        token_id = _lookup_token_id(tokenizer, "<|im_end|>")
        return [token_id] if token_id is not None else []


class Qwen3_5Spec(Qwen3FamilySpec):
    """Qwen3.5 maps to the "qwen3" oneof like the rest of the family, but its
    hybrid full-attention/linear-attention decoder layers need a dual cache
    (`Qwen3_5CausalLM.call_with_cache` expects a `(kv_cache, conv_cache,
    recurrent_cache)` tuple) that the LiteRT-LM adapter's single
    stacked-KV-tensor cache format cannot represent yet.
    """

    cache_structure = "hybrid"

    def describe_unsupported_cache_structure(self):
        return (
            "requires a 'hybrid' cache structure: Qwen3.5's hybrid "
            "full_attention/linear_attention layers use a dual cache "
            "structure (`call_with_cache` expects a `(kv_cache, conv_cache, "
            "recurrent_cache)` tuple, since linear-attention layers need a "
            "convolutional/recurrent state that a stacked KV tensor cannot "
            "represent), but the LiteRT-LM adapter only supports "
            '`cache_structure="single_stacked"` (a single stacked '
            "`[batch, num_layers, 2, cache_length, num_kv_heads, head_dim]` "
            "KV-cache tensor). Support for hybrid cache structures is not "
            "yet implemented."
        )


class Qwen2p5FamilySpec(LiteRTLMExportSpec):
    """Qwen and Qwen-MoE (pre-Qwen3 architecture) map to "qwen2p5"."""

    model_type = "qwen2p5"

    def get_chat_stop_token_ids(self, tokenizer):
        # Unlike Qwen3, `<|im_end|>` is not Qwen (2.5)'s registered
        # `end_token` (see `QwenTokenizer`/`QwenMoeTokenizer`, which use
        # `<|endoftext|>`), but ChatML-format instruct checkpoints may still
        # include `<|im_end|>` in their vocabulary as an ordinary token. Add
        # it as a chat-stop token when present; do nothing when it is not
        # (base/non-chat Qwen 2.5 vocabularies).
        token_id = _lookup_token_id(tokenizer, "<|im_end|>")
        return [token_id] if token_id is not None else []


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
    # Base Gemma (Gemma/Gemma2) has no dedicated ``LlmModelType`` subtype
    # (see ``GemmaSpec``), so this entry only exists to give it the shared
    # Gemma-family ``<end_of_turn>`` chat-stop-token behavior instead of
    # silently falling through to the plain ``LiteRTLMExportSpec`` default
    # (which has no chat-stop-token override).
    (
        "keras_hub.src.models.gemma.gemma_causal_lm",
        "GemmaCausalLM",
        GemmaSpec,
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
    # Qwen3MoeCausalLM above. It gets its own spec class (rather than
    # reusing Qwen3FamilySpec directly) because its hybrid cache structure
    # is not yet supported by the adapter -- see `Qwen3_5Spec`.
    (
        "keras_hub.src.models.qwen3_5.qwen3_5_causal_lm",
        "Qwen3_5CausalLM",
        Qwen3_5Spec,
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
    # Llama3CausalLM is a subclass of LlamaCausalLM, so its entry must come
    # first (registry order + isinstance first-match-wins) to get
    # ``Llama3Spec``'s ``<|eot_id|>`` chat-stop-token override instead of
    # falling through to the plain ``LlamaCausalLM`` entry below.
    (
        "keras_hub.src.models.llama3.llama3_causal_lm",
        "Llama3CausalLM",
        Llama3Spec,
    ),
    # NOTE: LlmModelType does not have a dedicated "llama" field; map Llama
    # checkpoints to generic_model so the protobuf oneof stays valid. (This
    # is also the ``LiteRTLMExportSpec`` default, so this entry only exists
    # to make the mapping explicit/greppable.) Base Llama (v1/v2) uses a
    # plain SentencePiece EOS with no secondary chat-stop token, so it does
    # not need its own spec class the way ``Llama3Spec`` does.
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
