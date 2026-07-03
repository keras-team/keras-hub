"""PyTorch adapter modules for exporting KerasHub CausalLM models to
LiteRT-LM."""

import contextlib
import inspect
import threading

import torch
from torch import nn

# Global lock serializing export-time mutations of PyTorch's default device.
# This keeps _cpu_default_device_scope thread-safe without changing semantics.
_DEFAULT_DEVICE_LOCK = threading.Lock()


@contextlib.contextmanager
def _cpu_default_device_scope():
    """Temporarily force PyTorch's default device to CPU.

    A module-level lock serializes this scope so concurrent exports (or
    exports running alongside GPU work) cannot observe a partially-applied
    default device.
    """
    with _DEFAULT_DEVICE_LOCK:
        original_device = torch.get_default_device()
        torch.set_default_device("cpu")
        try:
            yield
        finally:
            torch.set_default_device(original_device)


def _get_vision_encoder(backbone):
    """Return the vision encoder from a backbone, or ``None``."""
    return getattr(backbone, "vision_encoder", None) or getattr(
        backbone, "vit_encoder", None
    )


def _encoder_expects_single_image(vision_encoder):
    """Return ``True`` if the vision encoder takes one image at a time.

    Gemma3 accepts a batched stack of images with shape
    ``[B, N, H, W, 3]``. PaliGemma's ViT only accepts ``[B, H, W, 3]``. We
    detect this from the Functional model's input spec: a single-image
    encoder has one input whose shape (including the batch dimension) is
    4-D.
    """
    if not hasattr(vision_encoder, "inputs"):
        return False
    if len(vision_encoder.inputs) != 1:
        return False
    return len(vision_encoder.inputs[0].shape) == 4


def _run_vision_encoder(vision_encoder, images):
    """Run the vision encoder, reshaping inputs if necessary.

    For encoders that expect a single image per sample (e.g. PaliGemma),
    the LiteRT-LM runtime contract still passes ``[B, N, H, W, 3]``. We
    collapse the batch and image dimensions, run the encoder, and return
    features with shape ``[B * N, tokens_per_image, dim]``.
    """
    if not _encoder_expects_single_image(vision_encoder):
        out = vision_encoder(images)
    else:
        batch_size, num_images, height, width, channels = images.shape
        flat_images = images.reshape(
            batch_size * num_images, height, width, channels
        )
        out = vision_encoder(flat_images)
    return _extract_vision_features(out)


def _extract_vision_features(out):
    """Extract the feature tensor from a vision encoder output."""
    if isinstance(out, dict):
        features = out.get("features")
        if features is None:
            features = next(iter(out.values()))
        return features
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class KerasHubLiteRTAdapter(nn.Module):
    """Adapter that wraps a KerasHub CausalLM for LiteRT-LM export.

    The adapter exposes `forward_prefill` and `forward_decode` signatures
    compatible with `litert_torch.signature(...)`:

    Text-only inputs:
        tokens:       int32 [batch, seq_len]
        input_pos:    int32 [seq_len]   (position indices)
        kv_cache_k_0, kv_cache_v_0, ...: per-layer KV caches

    Multimodal prefill inputs (when model has a vision/audio encoder):
        images:       float32 [batch, num_images, H, W, 3]
        vision_indices: int32 [batch, num_vision_tokens]
        vision_mask:  int32 [batch, seq_len] or bool
        audio_mel:    float32 [batch, num_clips, num_frames, 128]
        audio_mel_mask: int32 [batch, num_clips, num_frames]
        audio_indices: int32 [batch, num_audio_tokens]
        audio_mask:   int32 [batch, seq_len] or bool
        (plus text inputs above)

    Outputs (prefill):
        kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches
        (no logits – LiteRT-LM extracts last-token logits via decode)

    Outputs (decode):
        logits:       float [batch, seq_len, vocab_size]
        kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches

    The adapter delegates to ``export_spec.stack_kv_cache``/
    ``unstack_kv_cache`` (see ``LiteRTLMExportSpec`` in ``model_specs.py``)
    to convert between the flat per-layer k/v tensors and whatever cache
    shape ``model.call_with_cache()`` expects for this family -- by default
    a single stacked ``[batch, num_layers, 2, cache_length, num_kv_heads,
    head_dim]`` tensor -- rather than hardcoding that shape here, so a
    future hybrid-cache family can override cache handling on its own spec
    instead of this adapter growing per-family branches.
    """

    def __init__(
        self,
        keras_model,
        num_layers,
        cache_length,
        export_spec,
        separate_vision_encoder=False,
    ):
        super().__init__()
        self.keras_model = keras_model
        self.num_layers = num_layers
        self.cache_length = cache_length
        self.separate_vision_encoder = separate_vision_encoder
        self.export_spec = export_spec
        self.cache_layout = export_spec.cache_layout

        vision_encoder = _get_vision_encoder(keras_model.backbone)
        self.has_vision = vision_encoder is not None
        # How this family's vision encoder consumes its input -- see
        # `LiteRTLMExportSpec.vision_input_style` in model_specs.py for the
        # three possible values ("raw_images" / "patch_values" /
        # "embedded_pixel_values"). `None` when the model has no vision
        # encoder at all.
        self.vision_input_style = (
            export_spec.vision_input_style if self.has_vision else None
        )
        # When exporting a separate vision encoder, keep the vision tower out of
        # the PREFILL_DECODE graph so its weights are not duplicated in the main
        # model. The cached `vision_input_style` still guides reshape logic.
        self.vision_encoder = (
            None if separate_vision_encoder else vision_encoder
        )

        self.has_audio = (
            hasattr(keras_model.backbone, "audio_encoder")
            and keras_model.backbone.audio_encoder is not None
        )

        # Cache the call_with_cache signature so we don't re-inspect it on every
        # forward pass during export tracing.
        call_params = set(
            inspect.signature(keras_model.call_with_cache).parameters.keys()
        )
        self._call_with_cache_params = call_params
        self._expects_input_features = "input_features" in call_params

    def forward_prefill(
        self,
        tokens,
        input_pos,
        images=None,
        vision_indices=None,
        vision_mask=None,
        pixel_values=None,
        pixel_position_ids=None,
        audio_mel=None,
        audio_mel_mask=None,
        audio_indices=None,
        audio_mask=None,
        mm_embedding=None,
        **kv_cache,
    ):
        """Prefill step – processes the full prompt at the given cache position.

        LiteRT-LM requires prefill to return **only** KV cache tensors
        (no logits).  The runtime extracts the last-token logits internally
        via a dedicated decode step.

        ``input_pos`` is a 1-D int32 tensor (e.g. ``[0, 1, 2, ...]`` for the
        first turn, or ``[N, N+1, ...]`` for subsequent turns).  The first
        element is used as the cache-update index so that prefill appends to
        the existing cache instead of overwriting from position 0.
        """
        cache = self.export_spec.stack_kv_cache(kv_cache, self.num_layers)
        # The first element of input_pos is the start position.
        cache_update_index = input_pos[0]

        img_embeddings, pixel_values_out = self._prepare_image_embeddings(
            tokens=tokens,
            images=images,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            mm_embedding=mm_embedding,
        )
        (
            audio_embeddings,
            input_features_out,
            input_features_mask_out,
        ) = self._prepare_audio_embeddings(
            audio_mel=audio_mel,
            audio_mel_mask=audio_mel_mask,
        )

        call_kwargs = self._build_call_with_cache_kwargs(
            img_embeddings=img_embeddings,
            vision_mask=vision_mask,
            vision_indices=vision_indices,
            audio_embeddings=audio_embeddings,
            audio_mask=audio_mask,
            audio_indices=audio_indices,
            pixel_values=pixel_values_out,
            input_features=input_features_out,
            input_features_mask=input_features_mask_out,
        )
        # Prefill returns only KV caches; LiteRT-LM extracts last-token logits
        # via a dedicated decode step.
        return self._call_with_cache(
            tokens, cache, cache_update_index, call_kwargs, return_logits=False
        )

    def _prepare_image_embeddings(
        self,
        tokens,
        images,
        pixel_values,
        pixel_position_ids,
        mm_embedding,
    ):
        """Return ``(img_embeddings, pixel_values_out)`` for prefill.

        Only one of the two return values is non-``None`` for a given model
        signature:

        - Gemma3n (``vision_input_style="embedded_pixel_values"``) expects
          raw ``pixel_values`` (returned as the second tuple item).
        - Separate-vision-encoder exports consume pre-computed
          ``mm_embedding``.
        - Gemma4 (``vision_input_style="patch_values"``) accepts
          preprocessed patch tensors.
        - Other vision encoders (``vision_input_style="raw_images"``, e.g.
          Gemma3, PaliGemma) accept raw ``images``.
        """
        if not self.has_vision:
            return None, None

        if self.vision_input_style == "embedded_pixel_values":
            # Gemma3n runs the vision encoder inside the backbone; pass the
            # raw preprocessed images through.
            return None, images

        if self.separate_vision_encoder:
            # Only Gemma4 needs a reshape here (see
            # ``Gemma4Spec.reshape_separate_vision_embeddings``); every other
            # family returns ``mm_embedding`` unchanged.
            reshape_fn = self.export_spec.reshape_separate_vision_embeddings
            img_embeddings = reshape_fn(
                mm_embedding, tokens, self.keras_model.preprocessor
            )
            return img_embeddings, None

        if self.vision_input_style == "patch_values":
            if pixel_values is not None and pixel_position_ids is not None:
                img_embeddings = self.vision_encoder(
                    {
                        "pixel_values": pixel_values,
                        "pixel_position_ids": pixel_position_ids,
                    }
                )
                return _extract_vision_features(img_embeddings), None
            return None, None

        if images is not None:
            return _run_vision_encoder(self.vision_encoder, images), None

        return None, None

    def _prepare_audio_embeddings(self, audio_mel, audio_mel_mask):
        """Return audio embeddings and optional input feature tensors."""
        if not self.has_audio or audio_mel is None:
            return None, None, None

        if self._expects_input_features:
            # Gemma3n runs the audio encoder inside the backbone.
            return None, audio_mel, audio_mel_mask

        audio_embeddings = self.keras_model.backbone.audio_encoder(
            audio_mel, audio_mel_mask
        )
        return audio_embeddings, None, None

    def forward_decode(self, tokens, input_pos, **kv_cache):
        """Decode step – processes a single token at *input_pos*.

        ``input_pos`` is a scalar int32 tensor (e.g. ``[3]``).  It is passed
        directly as the cache-update index so that the value remains a tensor
        inside the exported graph and is not baked in as a Python constant.
        """
        cache = self.export_spec.stack_kv_cache(kv_cache, self.num_layers)
        # Squeeze to a 0-D tensor so Keras cache operations receive a scalar.
        cache_update_index = input_pos.reshape(())
        call_kwargs = self._build_call_with_cache_kwargs(
            img_embeddings=None,
            vision_mask=None,
            vision_indices=None,
            audio_embeddings=None,
            audio_mask=None,
            audio_indices=None,
        )
        return self._call_with_cache(
            tokens, cache, cache_update_index, call_kwargs, return_logits=True
        )

    def _call_with_cache(
        self, tokens, cache, cache_update_index, call_kwargs, return_logits
    ):
        """Run ``keras_model.call_with_cache`` and return updated KV caches."""
        # Only Gemma3n forces an override here (a full-length padding mask;
        # see ``Gemma3nSpec.get_forced_call_with_cache_kwargs``); every other
        # family is a no-op.
        call_kwargs.update(
            self.export_spec.get_forced_call_with_cache_kwargs(
                tokens, self.cache_length
            )
        )
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens,
            cache,
            cache_update_index,
            **call_kwargs,
        )
        # Keras ops (``slice_update``/``scatter_update``, used internally by
        # every model's cache-update mechanism) are purely functional: they
        # never mutate their input in place, always returning a freshly
        # allocated tensor. `updated_cache` is therefore already guaranteed
        # to be independent of the `cache` argument above, without an extra
        # `.clone()` (verified empirically: `updated_cache.data_ptr() !=
        # cache.data_ptr()`, and mutating `updated_cache` in place does not
        # affect `cache`).
        outputs = self.export_spec.unstack_kv_cache(
            updated_cache, self.num_layers
        )
        if return_logits:
            outputs["logits"] = logits
        return outputs

    def _build_call_with_cache_kwargs(
        self,
        img_embeddings=None,
        vision_mask=None,
        vision_indices=None,
        audio_embeddings=None,
        audio_mask=None,
        audio_indices=None,
        pixel_values=None,
        input_features=None,
        input_features_mask=None,
    ):
        """Build kwargs dict for ``call_with_cache`` based on its signature."""
        params = self._call_with_cache_params
        values = {
            "img_embeddings": img_embeddings,
            "pixel_values": pixel_values,
            "vision_mask": vision_mask,
            "padding_mask": None,
            "vision_indices": vision_indices,
            "cache_update_mask": None,
            "audio_embeddings": audio_embeddings,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
            "audio_mask": audio_mask,
            "audio_indices": audio_indices,
        }
        return {k: v for k, v in values.items() if k in params}


class KerasHubVisionEncoderAdapter(nn.Module):
    """Adapter that wraps a KerasHub vision encoder for separate export.

    Gemma3 accepts raw ``images`` [B, N, H, W, 3]. Gemma4 accepts preprocessed
    patches via ``pixel_values`` and ``pixel_position_ids``. The output is
    always returned as a dictionary named ``features`` so that the LiteRT-LM
    signature matches upstream tensor names.
    """

    def __init__(self, keras_model):
        super().__init__()
        self.vision_encoder = _get_vision_encoder(keras_model.backbone)

    def forward(self, images=None, pixel_values=None, pixel_position_ids=None):
        if pixel_values is not None and pixel_position_ids is not None:
            out = self.vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                }
            )
        elif images is not None:
            out = _run_vision_encoder(self.vision_encoder, images)
        else:
            raise ValueError(
                "Vision encoder export requires either ``images`` or "
                "``pixel_values`` + ``pixel_position_ids``."
            )

        return {"features": _extract_vision_features(out)}


class KerasHubVisionAdapter(nn.Module):
    """No-op vision adapter exported as a separate LiteRT-LM model.

    KerasHub already projects vision features inside the vision encoder, so
    this adapter simply renames ``features`` to ``mm_embedding``.
    """

    def forward(self, features):
        return {"mm_embedding": features}
