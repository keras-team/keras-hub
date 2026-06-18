"""PyTorch adapter modules for exporting KerasHub CausalLM models to
LiteRT-LM."""

import contextlib
import inspect

import torch
from keras.src.backend.torch import core as torch_core
from torch import nn


class KerasHubLiteRTAdapter(nn.Module):
    """Adapter that wraps a KerasHub CausalLM for LiteRT-LM export.

    The adapter exposes `forward_prefill` and `forward_decode` signatures
    compatible with `litert_torch.signature(...)`:

    Text-only inputs:
        tokens:       int32 [batch, seq_len]
        input_pos:    int32 [seq_len]   (position indices)
        mask:         optional float mask (ignored by most models)
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

    The adapter stacks per-layer k/v tensors into the Keras cache format
    (``[batch, num_layers, 2, cache_length, num_kv_heads, head_dim]``),
    calls ``model.call_with_cache()``, and unstacks the result.
    """

    def __init__(
        self,
        keras_model,
        num_layers,
        cache_length,
        separate_vision_encoder=False,
    ):
        super().__init__()
        self.keras_model = keras_model
        self.num_layers = num_layers
        self.cache_length = cache_length
        self.separate_vision_encoder = separate_vision_encoder
        self.has_vision = (
            hasattr(keras_model.backbone, "vision_encoder")
            and keras_model.backbone.vision_encoder is not None
        )
        self.has_audio = (
            hasattr(keras_model.backbone, "audio_encoder")
            and keras_model.backbone.audio_encoder is not None
        )

    def forward_prefill(
        self,
        tokens,
        input_pos,
        mask=None,
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

        ``input_pos`` is a 1-D int64 tensor (e.g. ``[0, 1, 2, ...]`` for the
        first turn, or ``[N, N+1, ...]`` for subsequent turns).  The first
        element is used as the cache-update index so that prefill appends to
        the existing cache instead of overwriting from position 0.
        """
        cache = self._stack_kv_cache(kv_cache)
        # The first element of input_pos is the start position.
        cache_update_index = input_pos[0]

        # Run vision encoder if images are provided.
        img_embeddings = None
        if self.has_vision:
            if self.separate_vision_encoder:
                img_embeddings = mm_embedding
                # Gemma4 interleaves image embeddings with shape
                # (batch, num_images, tokens_per_image, hidden_dim). The separate
                # vision encoder/adapter produces a flat (batch*num_images, ...)
                # tensor, so reshape it back before passing to the language model.
                if img_embeddings is not None:
                    vision_encoder = self.keras_model.backbone.vision_encoder
                    is_gemma4_vision = (
                        hasattr(vision_encoder, "inputs")
                        and len(vision_encoder.inputs) == 2
                        and {inp.name for inp in vision_encoder.inputs}
                        == {"pixel_values", "pixel_position_ids"}
                    )
                    if is_gemma4_vision:
                        max_images = getattr(
                            self.keras_model.preprocessor,
                            "max_images_per_prompt",
                            1,
                        )
                        batch_size = tokens.shape[0]
                        img_embeddings = img_embeddings.reshape(
                            batch_size,
                            max_images,
                            img_embeddings.shape[1],
                            img_embeddings.shape[2],
                        )
            else:
                vision_encoder = self.keras_model.backbone.vision_encoder
                # Gemma4 vision encoder expects pixel_values and pixel_position_ids.
                # Detect via Functional model input names to avoid heavy imports.
                is_gemma4_vision = (
                    hasattr(vision_encoder, "inputs")
                    and len(vision_encoder.inputs) == 2
                    and {inp.name for inp in vision_encoder.inputs}
                    == {"pixel_values", "pixel_position_ids"}
                )
                if is_gemma4_vision:
                    if (
                        pixel_values is not None
                        and pixel_position_ids is not None
                    ):
                        img_embeddings = vision_encoder(
                            {
                                "pixel_values": pixel_values,
                                "pixel_position_ids": pixel_position_ids,
                            }
                        )
                elif images is not None:
                    img_embeddings = vision_encoder(images)

        # Run audio encoder if audio mel is provided.
        audio_embeddings = None
        if self.has_audio and audio_mel is not None:
            audio_embeddings = self.keras_model.backbone.audio_encoder(
                audio_mel, audio_mel_mask
            )

        call_kwargs = self._build_call_with_cache_kwargs(
            img_embeddings=img_embeddings,
            vision_mask=vision_mask,
            vision_indices=vision_indices,
            audio_embeddings=audio_embeddings,
            audio_mask=audio_mask,
            audio_indices=audio_indices,
        )
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens,
            cache,
            cache_update_index,
            **call_kwargs,
        )
        # Return ONLY the KV cache outputs (no logits).
        outputs = self._unstack_kv_cache(updated_cache)
        return outputs

    def forward_decode(self, tokens, input_pos, mask=None, **kv_cache):
        """Decode step – processes a single token at *input_pos*.

        ``input_pos`` is a scalar int32 tensor (e.g. ``[3]``).  It is passed
        directly as the cache-update index so that the value remains a tensor
        inside the exported graph and is not baked in as a Python constant.
        """
        cache = self._stack_kv_cache(kv_cache)
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
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens,
            cache,
            cache_update_index,
            **call_kwargs,
        )
        outputs = self._unstack_kv_cache(updated_cache)
        outputs["logits"] = logits
        return outputs

    def _stack_kv_cache(self, kv_cache):
        """Stack flat ``kv_cache_k_N`` / ``kv_cache_v_N`` into Keras format."""
        k_list = [kv_cache[f"kv_cache_k_{i}"] for i in range(self.num_layers)]
        v_list = [kv_cache[f"kv_cache_v_{i}"] for i in range(self.num_layers)]
        k_stack = torch.stack(k_list, dim=1)
        v_stack = torch.stack(v_list, dim=1)
        return torch.stack([k_stack, v_stack], dim=2)

    def _build_call_with_cache_kwargs(
        self,
        img_embeddings=None,
        vision_mask=None,
        vision_indices=None,
        audio_embeddings=None,
        audio_mask=None,
        audio_indices=None,
    ):
        """Build kwargs dict for ``call_with_cache`` based on its signature."""
        sig = inspect.signature(self.keras_model.call_with_cache)
        params = set(sig.parameters.keys())
        kwargs = {}
        if "img_embeddings" in params:
            kwargs["img_embeddings"] = img_embeddings
        if "vision_mask" in params:
            kwargs["vision_mask"] = vision_mask
        if "padding_mask" in params:
            kwargs["padding_mask"] = None
        if "vision_indices" in params:
            kwargs["vision_indices"] = vision_indices
        if "cache_update_mask" in params:
            kwargs["cache_update_mask"] = None
        if "audio_embeddings" in params:
            kwargs["audio_embeddings"] = audio_embeddings
        if "audio_mask" in params:
            kwargs["audio_mask"] = audio_mask
        if "audio_indices" in params:
            kwargs["audio_indices"] = audio_indices
        return kwargs

    def _unstack_kv_cache(self, cache):
        """Split Keras cache back into per-layer output tensors."""
        outputs = {}
        for i in range(self.num_layers):
            outputs[f"kv_cache_k_{i}"] = cache[:, i, 0, ...]
            outputs[f"kv_cache_v_{i}"] = cache[:, i, 1, ...]
        return outputs


class KerasHubVisionEncoderAdapter(nn.Module):
    """Adapter that wraps a KerasHub vision encoder for separate export.

    Gemma3 accepts raw ``images`` [B, N, H, W, 3]. Gemma4 accepts preprocessed
    patches via ``pixel_values`` and ``pixel_position_ids``. The output is
    always returned as a dictionary named ``features`` so that the LiteRT-LM
    signature matches upstream tensor names.
    """

    def __init__(self, keras_model):
        super().__init__()
        self.vision_encoder = keras_model.backbone.vision_encoder

    def forward(self, images=None, pixel_values=None, pixel_position_ids=None):
        if pixel_values is not None and pixel_position_ids is not None:
            out = self.vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                }
            )
        elif images is not None:
            out = self.vision_encoder(images)
        else:
            raise ValueError(
                "Vision encoder export requires either ``images`` or "
                "``pixel_values`` + ``pixel_position_ids``."
            )

        if isinstance(out, dict):
            features = out.get("features")
            if features is None:
                features = next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            features = out[0]
        else:
            features = out
        return {"features": features}


class KerasHubVisionAdapter(nn.Module):
    """No-op vision adapter exported as a separate LiteRT-LM model.

    KerasHub already projects vision features inside the vision encoder, so
    this adapter simply renames ``features`` to ``mm_embedding``.
    """

    def forward(self, features):
        return {"mm_embedding": features}


@contextlib.contextmanager
def _traceable_slice_update_scope():
    """Temporarily patch Keras torch-backend ``slice_update`` for torch.export.

    Keras ``ops.slice_update`` converts tensor ``start_indices`` to Python
    ints via ``.tolist()``, which fails during ``torch.export.export`` when
    the index is a dynamic tensor (e.g. the decode position).

    This patch replaces the implementation entirely for the export window:
    * Tensor starts (dynamic) with ``update_len == 1`` → ``index_copy_``.
    * All-Python-int starts (constant) → direct slice assignment without
      calling ``.item()`` or the original Keras logic.
    """
    original_slice_update = torch_core.slice_update
    original_slice = torch_core.slice

    def _patched_slice_update(inputs, start_indices, updates):
        inputs = torch_core.convert_to_tensor(inputs)
        updates = torch_core.convert_to_tensor(updates)

        # Normalise start_indices to a list while preserving types.
        if isinstance(start_indices, (list, tuple)):
            starts = list(start_indices)
        else:
            start_indices = torch_core.convert_to_tensor(
                start_indices, dtype="int64"
            )
            starts = [
                start_indices.reshape(-1)[i]
                for i in range(start_indices.numel())
            ]

        # Dimensions whose start is a tensor → potentially dynamic.
        tensor_dims = []
        for dim, start in enumerate(starts):
            if isinstance(start, torch.Tensor):
                tensor_dims.append(dim)

        # Single dynamic dimension → loop over index_copy_ (works for any
        # update_len because update_len is a constant at export time).
        if len(tensor_dims) == 1:
            dim = tensor_dims[0]
            update_len = updates.shape[dim]
            start = starts[dim].reshape(())

            outputs = torch.clone(inputs)
            for i in range(update_len):
                index = (start + i).reshape(()).unsqueeze(0)
                update_slice = updates.select(dim, i).unsqueeze(dim)
                outputs.index_copy_(dim, index, update_slice)
            return outputs

        if len(tensor_dims) > 1:
            raise RuntimeError(
                "slice_update patch does not support multiple dynamic start "
                f"indices. Dynamic dims: {tensor_dims}."
            )

        # All starts are plain Python ints → direct slice assignment.
        # We avoid the original Keras implementation because it converts
        # the starts to a tensor and calls ``.item()``, which breaks under
        # ``torch.export`` on older Keras versions (< 3.15).
        outputs = torch.clone(inputs)
        slices = []
        for start, size in zip(starts, updates.shape):
            slices.append(slice(start, start + size))
        outputs[slices] = updates
        return outputs

    def _patched_slice(inputs, start_indices, shape):
        inputs = torch_core.convert_to_tensor(inputs)

        # Normalise start_indices and shape to lists while preserving types.
        if isinstance(start_indices, (list, tuple)):
            starts = list(start_indices)
        else:
            start_indices = torch_core.convert_to_tensor(
                start_indices, dtype="int64"
            )
            starts = [
                start_indices.reshape(-1)[i]
                for i in range(start_indices.numel())
            ]

        if isinstance(shape, (list, tuple)):
            lengths = list(shape)
        else:
            shape = torch_core.convert_to_tensor(shape, dtype="int64")
            lengths = [shape.reshape(-1)[i] for i in range(shape.numel())]

        # Dimensions whose start is a tensor → potentially dynamic.
        tensor_dims = []
        for dim, start in enumerate(starts):
            if isinstance(start, torch.Tensor):
                tensor_dims.append(dim)

        # No dynamic starts → fall back to the original implementation.
        if len(tensor_dims) == 0:
            return original_slice(inputs, starts, lengths)

        # Single dynamic dimension → use index_select to avoid unbacked
        # symbols in torch.export.
        if len(tensor_dims) == 1:
            dim = tensor_dims[0]
            start = starts[dim].reshape(())
            length = lengths[dim]

            # Build the selection indices dynamically.
            if isinstance(length, int):
                indices = torch.arange(
                    length, dtype=torch.int64, device=inputs.device
                )
            else:
                # length is a SymInt or tensor – try arange with it.
                indices = torch.arange(
                    length, dtype=torch.int64, device=inputs.device
                )
            indices = indices + start
            result = torch.index_select(inputs, dim, indices)

            # Apply static slicing for the remaining dimensions.
            for d, (s, l) in enumerate(zip(starts, lengths)):
                if d != dim and (s != 0 or l != result.shape[d]):
                    result = torch.narrow(result, d, s, l)
            return result

        # Multiple dynamic dimensions – fall back (will likely fail in export).
        return original_slice(inputs, starts, lengths)

    torch_core.slice_update = _patched_slice_update
    torch_core.slice = _patched_slice
    try:
        yield
    finally:
        torch_core.slice_update = original_slice_update
        torch_core.slice = original_slice
