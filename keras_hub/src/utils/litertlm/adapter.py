"""PyTorch adapters for exporting KerasHub CausalLM models to LiteRT-LM."""

import contextlib
import inspect
import threading

import torch
from torch import nn

from keras_hub.src.utils.litertlm.model_specs import _get_vision_encoder

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


def _run_vision_encoder(vision_encoder, images, flatten_image_batch):
    """Run the vision encoder, reshaping inputs if necessary.

    For single-image encoders (``flatten_image_batch=True``, e.g.
    PaliGemma), collapse the runtime's ``[B, N, H, W, 3]`` stack to
    ``[B * N, H, W, 3]`` and return ``[B * N, tokens_per_image, dim]``.
    """
    if not flatten_image_batch:
        out = vision_encoder(images)
    else:
        batch_size, num_images, height, width, channels = images.shape
        flat_images = images.reshape(
            batch_size * num_images, height, width, channels
        )
        out = vision_encoder(flat_images)
    return out


def _checked_vision_features(out):
    """Return a vision encoder's output, rejecting non-tensor outputs."""
    if not isinstance(out, torch.Tensor):
        raise ValueError(
            "The vision encoder must return a single feature tensor for "
            "LiteRT-LM export. Received: "
            f"out={type(out).__module__}.{type(out).__name__}. Wrap encoders "
            "that return a dict or a tuple so they return the feature tensor."
        )
    return out


def _vision_style_mismatch_message(
    style,
    expected,
    supplied_images,
    supplied_pixel_values,
    supplied_pixel_position_ids,
):
    """Build the error for a vision_input_style/supplied-args mismatch."""
    supplied = [
        name
        for name, present in (
            ("images", supplied_images),
            ("pixel_values", supplied_pixel_values),
            ("pixel_position_ids", supplied_pixel_position_ids),
        )
        if present
    ]
    supplied_str = ", ".join(supplied) if supplied else "none"
    return (
        f"vision_input_style={style!r} requires {expected} as input. "
        f"Received vision inputs: {supplied_str}. Check the model's "
        "`LiteRTLMExportSpec.vision_input_style`."
    )


def _run_vision_encoder_for_style(
    vision_encoder,
    vision_input_style,
    flatten_image_batch,
    images,
    pixel_values,
    pixel_position_ids,
):
    """Validate vision inputs against the declared style and run the encoder.

    Callers must only pass the two encoder-run styles
    (``"patch_values"`` / ``"raw_images"``).
    """
    if vision_input_style == "patch_values":
        if pixel_values is None or pixel_position_ids is None:
            raise ValueError(
                _vision_style_mismatch_message(
                    style="patch_values",
                    expected="`pixel_values` and `pixel_position_ids`",
                    supplied_images=images is not None,
                    supplied_pixel_values=pixel_values is not None,
                    supplied_pixel_position_ids=pixel_position_ids is not None,
                )
            )
        return _checked_vision_features(
            vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                }
            )
        )
    if images is None:
        raise ValueError(
            _vision_style_mismatch_message(
                style="raw_images",
                expected="`images`",
                supplied_images=images is not None,
                supplied_pixel_values=pixel_values is not None,
                supplied_pixel_position_ids=pixel_position_ids is not None,
            )
        )
    return _checked_vision_features(
        _run_vision_encoder(vision_encoder, images, flatten_image_batch)
    )


class KerasHubLiteRTAdapter(nn.Module):
    """Adapter that wraps a KerasHub CausalLM for LiteRT-LM export.

    The adapter exposes `forward_prefill` and `forward_decode` signatures
    compatible with `litert_torch.signature(...)`:

    Text-only inputs:
        tokens:       int32 [batch, seq_len]
        input_pos:    int32 [seq_len]   (position indices)
        kv_cache_k_0, kv_cache_v_0, ...: per-layer KV caches

    Multimodal prefill inputs (when the model has a vision encoder):
        images:       float32 [batch, num_images, H, W, 3]
        pixel_values: float32 [batch, num_images, num_patches, patch_dim]
        pixel_position_ids: int32 [batch, num_images, num_patches]
        vision_indices: int32 [batch, num_vision_tokens]
        vision_mask:  int32 [batch, seq_len] or bool
        (plus text inputs above)

    Outputs (prefill):
        kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches
        (no logits -- LiteRT-LM extracts last-token logits via decode)

    Outputs (decode):
        logits:       float [batch, seq_len, vocab_size]
        kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches

    The adapter delegates to ``export_spec.stack_kv_cache``/
    ``unstack_kv_cache`` to convert between the flat per-layer k/v tensors
    and the cache shape ``model.call_with_cache()`` expects for this family.
    """

    def __init__(
        self,
        keras_model,
        num_layers,
        cache_length,
        export_spec,
    ):
        super().__init__()
        self.keras_model = keras_model
        self.num_layers = num_layers
        self.cache_length = cache_length
        self.export_spec = export_spec

        vision_encoder = _get_vision_encoder(keras_model.backbone)
        self.has_vision = vision_encoder is not None
        # The family's declared `vision_input_style`, or `None` when the
        # model has no vision encoder.
        self.vision_input_style = (
            export_spec.vision_input_style if self.has_vision else None
        )
        # The family's declared `flatten_image_batch` (single-image ViT),
        # or `False` when the model has no vision encoder.
        self.flatten_image_batch = (
            export_spec.flatten_image_batch if self.has_vision else False
        )
        self.vision_encoder = vision_encoder

        # Cache the call_with_cache signature so we don't re-inspect it on every
        # forward pass during export tracing.
        call_params = set(
            inspect.signature(keras_model.call_with_cache).parameters.keys()
        )
        self._call_with_cache_params = call_params

    def forward_prefill(
        self,
        tokens,
        input_pos,
        images=None,
        vision_indices=None,
        vision_mask=None,
        pixel_values=None,
        pixel_position_ids=None,
        **kv_cache,
    ):
        """Prefill step: process the full prompt at the given cache position.

        LiteRT-LM requires prefill to return **only** KV cache tensors
        (no logits). The runtime extracts the last-token logits internally
        via a dedicated decode step.

        ``input_pos`` is a 1-D int32 tensor (e.g. ``[0, 1, 2, ...]`` for the
        first turn, or ``[N, N+1, ...]`` for subsequent turns). The first
        element is used as the cache-update index so that prefill appends to
        the existing cache instead of overwriting from position 0.
        """
        cache = self.export_spec.stack_kv_cache(kv_cache, self.num_layers)
        # The first element of input_pos is the start position.
        cache_update_index = input_pos[0]

        img_embeddings, pixel_values_out = self._prepare_image_embeddings(
            images=images,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
        )

        call_kwargs = self._build_call_with_cache_kwargs(
            img_embeddings=img_embeddings,
            vision_mask=vision_mask,
            vision_indices=vision_indices,
            pixel_values=pixel_values_out,
        )
        return self._call_with_cache(
            tokens, cache, cache_update_index, call_kwargs, return_logits=False
        )

    def _prepare_image_embeddings(
        self,
        images,
        pixel_values,
        pixel_position_ids,
    ):
        """Return ``(img_embeddings, pixel_values_out)`` for prefill.

        Only one of the two return values is non-``None`` for a given model
        signature:

        - Gemma3n (``vision_input_style="embedded_pixel_values"``) expects
          raw ``pixel_values`` (returned as the second tuple item).
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

        if self.vision_input_style in ("patch_values", "raw_images"):
            return (
                _run_vision_encoder_for_style(
                    self.vision_encoder,
                    self.vision_input_style,
                    self.flatten_image_batch,
                    images=images,
                    pixel_values=pixel_values,
                    pixel_position_ids=pixel_position_ids,
                ),
                None,
            )

        # has_vision is True and none of the known styles matched -- a spec
        # declared a vision_input_style the adapter does not handle. Fail
        # loudly rather than silently returning no embeddings.
        raise ValueError(
            "Unhandled vision_input_style "
            f"{self.vision_input_style!r} for a model with a vision encoder. "
            "Expected one of 'raw_images', 'patch_values', "
            "'embedded_pixel_values'."
        )

    def forward_decode(self, tokens, input_pos, **kv_cache):
        """Decode step: process a single token at ``input_pos``.

        ``input_pos`` is a scalar int32 tensor (e.g. ``[3]``). It is passed
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
        # Keras cache-update ops (`slice_update`/`scatter_update`) are
        # purely functional, so `updated_cache` is already a fresh,
        # non-aliased tensor -- no `.clone()` needed.
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
        pixel_values=None,
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
        }
        return {k: v for k, v in values.items() if k in params}
