"""PyTorch adapter modules for exporting KerasHub CausalLM models to
LiteRT-LM."""

import contextlib
import inspect
import threading
import unittest.mock

import torch
from keras.src.backend.torch import core as torch_core
from keras.src.backend.torch import numpy as torch_backend_numpy
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


def _is_gemma4_vision_encoder(vision_encoder):
    """Return ``True`` if *vision_encoder* uses Gemma4 patch inputs."""
    return (
        hasattr(vision_encoder, "inputs")
        and len(vision_encoder.inputs) == 2
        and {inp.name for inp in vision_encoder.inputs}
        == {"pixel_values", "pixel_position_ids"}
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
        cache_layout="standard",
    ):
        super().__init__()
        self.keras_model = keras_model
        self.num_layers = num_layers
        self.cache_length = cache_length
        self.separate_vision_encoder = separate_vision_encoder
        self.cache_layout = cache_layout

        vision_encoder = _get_vision_encoder(keras_model.backbone)
        self.has_vision = vision_encoder is not None
        self.is_gemma4_vision = (
            vision_encoder is not None
            and _is_gemma4_vision_encoder(vision_encoder)
        )
        # When exporting a separate vision encoder, keep the vision tower out of
        # the PREFILL_DECODE graph so its weights are not duplicated in the main
        # model. The cached `is_gemma4_vision` flag still guides reshape logic.
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
        self._expects_pixel_values = "pixel_values" in call_params
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
        cache = self._stack_kv_cache(kv_cache)
        # The first element of input_pos is the start position.
        cache_update_index = input_pos[0]

        # Run vision encoder if images are provided.
        img_embeddings = None
        pixel_values_out = None
        if self.has_vision:
            if self._expects_pixel_values:
                # Gemma3n runs the vision encoder inside the backbone; pass the
                # raw preprocessed images through.
                pixel_values_out = images
            elif self.separate_vision_encoder:
                img_embeddings = mm_embedding
                # Gemma4 interleaves image embeddings with shape
                # (batch, num_images, tokens_per_image, hidden_dim).
                # The separate vision encoder/adapter produces a flat
                # (batch*num_images, ...) tensor, so reshape it back before
                # passing to the language model.
                if img_embeddings is not None and self.is_gemma4_vision:
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
            elif self.is_gemma4_vision:
                if pixel_values is not None and pixel_position_ids is not None:
                    img_embeddings = self.vision_encoder(
                        {
                            "pixel_values": pixel_values,
                            "pixel_position_ids": pixel_position_ids,
                        }
                    )
            elif images is not None:
                img_embeddings = _run_vision_encoder(
                    self.vision_encoder, images
                )

        # Run audio encoder if audio mel is provided.
        audio_embeddings = None
        input_features_out = None
        input_features_mask_out = None
        if self.has_audio and audio_mel is not None:
            if self._expects_input_features:
                # Gemma3n runs the audio encoder inside the backbone.
                input_features_out = audio_mel
                input_features_mask_out = audio_mel_mask
            else:
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
            pixel_values=pixel_values_out,
            input_features=input_features_out,
            input_features_mask=input_features_mask_out,
        )
        # Prefill returns only KV caches; LiteRT-LM extracts last-token logits
        # via a dedicated decode step.
        return self._call_with_cache(
            tokens, cache, cache_update_index, call_kwargs, return_logits=False
        )

    def forward_decode(self, tokens, input_pos, **kv_cache):
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
        return self._call_with_cache(
            tokens, cache, cache_update_index, call_kwargs, return_logits=True
        )

    def _call_with_cache(
        self, tokens, cache, cache_update_index, call_kwargs, return_logits
    ):
        """Run ``keras_model.call_with_cache`` and return updated KV caches."""
        if self.cache_layout == "gemma3n":
            # Gemma3n's attention mask computation requires the padding mask
            # to span the full cache length, otherwise a seq_len shorter than
            # cache_length causes a broadcasting error between the causal and
            # padding masks. During export we always pass full-length valid
            # tokens, so a ones mask of cache length is correct.
            call_kwargs["padding_mask"] = torch.ones(
                (tokens.shape[0], self.cache_length),
                dtype=torch.bool,
                device=tokens.device,
            )
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens,
            cache,
            cache_update_index,
            **call_kwargs,
        )
        # Clone the updated cache before unstacking so that TFLite does not
        # alias the returned KV-cache outputs with activation buffers.
        outputs = self._unstack_kv_cache(updated_cache.clone())
        if return_logits:
            outputs["logits"] = logits
        return outputs

    def _stack_kv_cache(self, kv_cache):
        """Stack flat ``kv_cache_k_N`` / ``kv_cache_v_N`` into Keras format.

        The returned tensor is cloned so that downstream in-place cache
        updates do not corrupt the input/output buffers that TFLite may
        alias.
        """
        k_list = [kv_cache[f"kv_cache_k_{i}"] for i in range(self.num_layers)]
        v_list = [kv_cache[f"kv_cache_v_{i}"] for i in range(self.num_layers)]
        k_stack = torch.stack(k_list, dim=1)
        v_stack = torch.stack(v_list, dim=1)
        return torch.stack([k_stack, v_stack], dim=2).clone()

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
        kwargs = {}
        if "img_embeddings" in params:
            kwargs["img_embeddings"] = img_embeddings
        if "pixel_values" in params:
            kwargs["pixel_values"] = pixel_values
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
        if "input_features" in params:
            kwargs["input_features"] = input_features
        if "input_features_mask" in params:
            kwargs["input_features_mask"] = input_features_mask
        if "audio_mask" in params:
            kwargs["audio_mask"] = audio_mask
        if "audio_indices" in params:
            kwargs["audio_indices"] = audio_indices
        return kwargs

    def _unstack_kv_cache(self, cache):
        """Split Keras cache back into per-layer output tensors.

        Each slice is cloned so that TFLite cannot alias the returned KV
        cache tensors with intermediate activation buffers. LiteRT-LM
        allocates dedicated output buffers for these tensors, so the clone
        is only a trace-time guard against aliasing in the exported graph.
        """
        outputs = {}
        for i in range(self.num_layers):
            outputs[f"kv_cache_k_{i}"] = cache[:, i, 0, ...].clone()
            outputs[f"kv_cache_v_{i}"] = cache[:, i, 1, ...].clone()
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


def _normalize_start_indices(start_indices):
    """Convert ``start_indices`` to a list preserving tensor elements."""
    if isinstance(start_indices, (list, tuple)):
        return list(start_indices)
    start_indices = torch_core.convert_to_tensor(start_indices, dtype="int64")
    if start_indices.ndim != 1:
        raise ValueError(
            "`start_indices` must be a 1-D tensor or a list/tuple of ints. "
            f"Received shape: {tuple(start_indices.shape)}."
        )
    return list(start_indices.reshape(-1).unbind())


def _make_patched_slice(original_slice):
    """Return a traceable ``slice`` replacement."""

    def _patched_slice(inputs, start_indices, shape):
        inputs = torch_core.convert_to_tensor(inputs)

        starts = _normalize_start_indices(start_indices)

        if isinstance(shape, (list, tuple)):
            lengths = list(shape)
        else:
            shape = torch_core.convert_to_tensor(shape, dtype="int64")
            lengths = list(shape.reshape(-1).unbind())

        def _is_dynamic(value):
            # ``torch.SymInt`` values are not plain Python ints and require
            # tensor-based slicing to avoid data-dependent guards.
            return isinstance(value, torch.Tensor) or isinstance(
                value, torch.SymInt
            )

        # Dimensions whose start or length is dynamic.
        dynamic_dims = [
            dim
            for dim, (start, length) in enumerate(zip(starts, lengths))
            if _is_dynamic(start) or _is_dynamic(length)
        ]

        # No dynamic values → use Python slice objects directly.
        if len(dynamic_dims) == 0:
            slices = tuple(
                slice(start, start + length)
                for start, length in zip(starts, lengths)
            )
            return inputs[slices]

        # Single dynamic dimension → build indices with ``torch.arange`` and
        # use ``index_select``. This keeps the output shape symbolic and avoids
        # unbacked symbols that ``torch.export`` cannot resolve.
        if len(dynamic_dims) == 1:
            dim = dynamic_dims[0]
            start = starts[dim]
            if not isinstance(start, torch.Tensor):
                start = torch_core.convert_to_tensor(
                    start, dtype="int32", device=inputs.device
                )
            start = start.reshape(())
            length = lengths[dim]

            indices = torch.arange(
                length, dtype=torch.int32, device=inputs.device
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

    return _patched_slice


@contextlib.contextmanager
def _traceable_slice_update_scope():
    """Temporarily patch Keras torch-backend ``slice`` for torch.export.

    Keras ``ops.slice`` can introduce unbacked symbols during
    ``torch.export.export`` for dynamic start/length values. This patch keeps
    slicing traceable for the common single-dynamic-dimension case.

    Uses ``unittest.mock.patch.object`` so restoration is reliable even when
    an exception escapes.
    """
    original_slice = torch_core.slice
    with unittest.mock.patch.object(
        torch_core,
        "slice",
        _make_patched_slice(original_slice),
    ):
        yield


def _traceable_dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None,
):
    """Traceable replacement for Keras torch-backend ``dot_product_attention``.

    ``torch.nn.functional.scaled_dot_product_attention`` lowers to fused ATen
    ops such as ``aten._scaled_dot_product_efficient_attention`` that
    ``litert_torch`` cannot translate to TFLite.  This implementation expands
    attention to a plain ``matmul`` + ``softmax`` + ``matmul`` sequence that
    ``litert_torch`` handles well.

    The function mirrors Keras's ``dot_product_attention`` signature and shape
    convention: inputs are ``[batch, seq_len, num_heads, head_dim]`` and the
    output is returned in the same layout.
    """
    from keras.src import backend

    del flash_attention  # Fused flash attention is not exportable.

    query = torch_core.convert_to_tensor(query)
    key = torch_core.convert_to_tensor(key)
    value = torch_core.convert_to_tensor(value)

    compute_dtype = backend.result_type(query.dtype, key.dtype, value.dtype)
    query = torch_core.cast(query, compute_dtype)
    key = torch_core.cast(key, compute_dtype)
    value = torch_core.cast(value, compute_dtype)

    if scale is None:
        scale = float(query.shape[-1]) ** -0.5
    scale = torch_core.convert_to_tensor(scale, dtype=compute_dtype)

    if mask is not None:
        mask = torch_core.convert_to_tensor(mask, dtype="bool")
        if is_causal:
            q_len, kv_len = query.shape[1], key.shape[1]
            causal_mask = torch.tril(
                torch.ones(
                    (q_len, kv_len), dtype=torch.bool, device=mask.device
                )
            )
            mask = torch.logical_and(mask, causal_mask)
        is_causal = False
    elif is_causal:
        q_len, kv_len = query.shape[1], key.shape[1]
        mask = torch.tril(
            torch.ones((q_len, kv_len), dtype=torch.bool, device=query.device)
        )

    # Move heads to the batch dimension to match SDPA's score layout
    # [batch, num_heads, seq_len, head_dim].
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if bias is not None:
        scores = scores + torch_core.convert_to_tensor(
            bias, dtype=compute_dtype
        )

    if mask is not None:
        large_neg = torch.tensor(
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        )
        scores = torch.where(mask, scores, large_neg)

    if attn_logits_soft_cap is not None:
        cap = torch_core.convert_to_tensor(
            attn_logits_soft_cap, dtype=compute_dtype
        )
        scores = torch.tanh(scores / cap) * cap

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    return output.transpose(1, 2)


@contextlib.contextmanager
def _traceable_dot_product_attention_scope():
    """Temporarily patch Keras torch-backend ``dot_product_attention``.

    Uses ``unittest.mock.patch.object`` so restoration is reliable even when
    an exception escapes.
    """
    from keras.src.backend.torch import nn as torch_backend_nn

    with unittest.mock.patch.object(
        torch_backend_nn,
        "dot_product_attention",
        _traceable_dot_product_attention,
    ):
        yield


def _patched_one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    """Traceable replacement for Keras torch-backend ``one_hot``.

    ``torch.nn.functional.one_hot`` inserts runtime assertions that class
    values are non-negative. Under ``torch.export`` these become
    ``aten._assert_async.msg`` ops, which ``litert_torch`` cannot lower.

    This implementation uses equality against ``torch.arange``, which produces
    the same result for non-negative indices and does not introduce
    unlowerable assertions. Negative indices are preserved as all-zero vectors,
    matching the original behavior.

    Integer tensors are kept in int32 so the exported MLIR remains compatible
    with ``litert_torch``'s i32-based TFLite lowering.
    """
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with torch backend")
    x = torch_core.convert_to_tensor(x, dtype=torch.int32)
    x_clamped = torch.clamp(x, min=0)
    output = x_clamped.unsqueeze(-1) == torch.arange(
        num_classes, dtype=torch.int32, device=x.device
    )
    # Preserve original behavior for negative indices.
    zero = torch.zeros_like(output)
    output = torch.where(x.unsqueeze(-1) >= 0, output, zero)
    if dtype is None:
        dtype = "float32"
    output = torch_core.convert_to_tensor(output, dtype=dtype)
    dims = output.dim()
    if axis < 0:
        axis = dims + axis
    if axis < 0 or axis >= dims:
        raise ValueError(
            f"`axis` {axis} is out of bounds for one-hot output with "
            f"{dims} dimensions."
        )
    if axis != dims - 1:
        new_axes_order = list(range(dims))
        new_axes_order[axis] = dims - 1
        for ax in range(axis + 1, dims):
            new_axes_order[ax] -= 1
        output = output.permute(new_axes_order)
    return output


@contextlib.contextmanager
def _traceable_one_hot_scope():
    """Temporarily patch Keras torch-backend ``one_hot`` for torch.export.

    Uses ``unittest.mock.patch.object`` so restoration is reliable even when
    an exception escapes.
    """
    from keras.src.backend.torch import nn as torch_backend_nn

    with unittest.mock.patch.object(
        torch_backend_nn, "one_hot", _patched_one_hot
    ):
        yield


@contextlib.contextmanager
def _traceable_arange_scope():
    """Patch Keras torch-backend ``arange`` to default integer ranges to int32.

    ``torch.arange`` returns int64 for integer arguments. ``litert_torch``'s
    i64-to-i32 conversion pass does not always propagate through nested
    ``func.call`` boundaries, so integer ranges produced inside the model
    (e.g. position-embedding indices) must be int32 from the start.
    """
    from keras.src.backend.common import dtypes as keras_dtypes

    original_arange = torch_backend_numpy.arange

    def _patched_arange(start, stop=None, step=None, dtype=None):
        if dtype is None:
            dtypes_to_resolve = [
                getattr(start, "dtype", type(start)),
            ]
            if stop is not None:
                dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
            if step is not None:
                dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
            resolved = keras_dtypes.result_type(*dtypes_to_resolve)
            if str(resolved).startswith("int"):
                dtype = torch.int32
        return original_arange(start, stop=stop, step=step, dtype=dtype)

    with unittest.mock.patch.object(
        torch_backend_numpy, "arange", _patched_arange
    ):
        yield


@contextlib.contextmanager
def _traceable_take_scope():
    """Patch Keras torch-backend ``take`` to keep embedding indices as int32.

    The default implementation casts integer indices to int64 before calling
    ``torch.nn.functional.embedding``. ``litert_torch``'s TFLite embedding
    lowering expects int32 indices consistent with the traced function
    signature, so we keep indices in int32 for the embedding-lookup case.
    """

    def _patched_take(x, indices, axis=None):
        x = torch_core.convert_to_tensor(x)
        indices = torch_core.convert_to_tensor(indices, dtype=torch.int32)
        x_dim = x.shape[axis] if axis is not None else x.shape[0]
        indices = torch.where(
            indices < 0,
            indices + x_dim,
            indices,
        )
        if x.ndim == 2 and axis == 0:
            return torch.nn.functional.embedding(indices, x)
        if axis is None:
            x = torch.reshape(x, (-1,))
            axis = 0
        axis = torch_backend_numpy.canonicalize_axis(axis, x.ndim)
        shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
        indices = indices.ravel()
        out = torch.index_select(x, dim=axis, index=indices).squeeze(axis)
        return out.reshape(shape)

    with unittest.mock.patch.object(torch_backend_numpy, "take", _patched_take):
        yield


@contextlib.contextmanager
def _traceable_scatter_update_scope():
    """Patch Keras torch-backend ``scatter_update`` to keep indices int32.

    The default implementation casts indices to int64. ``litert_torch``'s
    TFLite scatter lowering expects int32 indices, so we keep them in int32
    during export.
    """

    def _patched_scatter_update(inputs, indices, updates, reduction=None):
        inputs = torch_core.convert_to_tensor(inputs)
        indices = torch_core.convert_to_tensor(indices, dtype=torch.int32)
        updates = torch_core.convert_to_tensor(updates, dtype=inputs.dtype)
        indices = torch.transpose(indices, 0, 1)
        idx = tuple(indices)

        outputs = torch.clone(inputs)
        if reduction is None:
            outputs[idx] = updates
        elif reduction == "add":
            outputs.index_put_(idx, updates, accumulate=True)
        elif reduction == "max":
            indices_t = indices.T
            for i in range(indices_t.shape[0]):
                idx = tuple(indices_t[i])
                outputs[idx] = torch.maximum(outputs[idx], updates[i])
        elif reduction == "min":
            indices_t = indices.T
            for i in range(indices_t.shape[0]):
                idx = tuple(indices_t[i])
                outputs[idx] = torch.minimum(outputs[idx], updates[i])
        elif reduction == "mul":
            indices_t = indices.T
            for i in range(indices_t.shape[0]):
                idx = tuple(indices_t[i])
                outputs[idx] = outputs[idx] * updates[i]
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return outputs

    with unittest.mock.patch.object(
        torch_core, "scatter_update", _patched_scatter_update
    ):
        yield


def _is_scalar_integer(value):
    """Return ``True`` if *value* is a scalar integer (Python or numpy)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    # Accept 0-D numpy integer arrays / tensors with a single integer value.
    if hasattr(value, "dtype") and hasattr(value, "ndim"):
        return value.ndim == 0 and "int" in str(value.dtype)
    return False


def _traceable_repeat(x, repeats, axis=None):
    """Traceable replacement for Keras torch-backend ``numpy.repeat``.

    ``torch.repeat_interleave`` lowers to ``aten.repeat_interleave.Tensor``,
    which ``litert_torch`` cannot translate to TFLite. For the common case of
    repeating a tensor by a scalar integer along a single axis (e.g. GQA
    key/value head repetition), this implementation uses
    ``unsqueeze + expand + reshape``, which ``litert_torch`` handles.
    """
    x = torch_core.convert_to_tensor(x)

    if axis is not None and _is_scalar_integer(repeats):
        repeats = int(repeats)
        if repeats < 0:
            raise ValueError("`repeats` must be non-negative.")
        if repeats == 1:
            return x
        if axis < 0:
            axis = x.ndim + axis
        shape = list(x.shape)
        x = x.unsqueeze(axis + 1)
        expand_shape = [-1] * x.ndim
        expand_shape[axis + 1] = repeats
        x = x.expand(expand_shape)
        new_shape = list(shape)
        new_shape[axis] = shape[axis] * repeats
        return x.reshape(new_shape)

    # Fall back to the original implementation for list/tuple repeats or
    # dynamic repeat counts.
    return torch_backend_numpy.repeat(x, repeats, axis=axis)


@contextlib.contextmanager
def _traceable_repeat_scope():
    """Temporarily patch Keras torch-backend ``repeat`` for torch.export.

    Uses ``unittest.mock.patch.object`` so restoration is reliable even when
    an exception escapes.
    """
    with unittest.mock.patch.object(
        torch_backend_numpy, "repeat", _traceable_repeat
    ):
        yield
