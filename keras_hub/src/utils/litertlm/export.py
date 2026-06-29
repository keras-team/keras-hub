"""Export KerasHub CausalLM models to LiteRT-LM `.litertlm` bundles."""

import contextlib
import importlib.util
import inspect
import os
import tempfile

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
from keras_hub.src.utils.preset_utils import TOKENIZER_ASSET_DIR

# Special token strings used when populating vision/audio metadata. Keeping
# them as named constants makes it easy to audit which tokens each model
# family expects and avoids scattering literals through the metadata helpers.
_GEMMA3_START_OF_IMAGE_TOKEN = "<start_of_image>"
_GEMMA3_END_OF_IMAGE_TOKEN = "<end_of_image>"
_GEMMA4_START_OF_IMAGE_TOKEN = "<|image>"
_GEMMA4_END_OF_IMAGE_TOKEN = "<image|>"
_AUDIO_START_TOKEN = "<|audio>"
_AUDIO_END_TOKEN = "<audio|>"

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

# (module_path, class_name, model_type). Imported lazily inside
# ``_detect_llm_model_type`` to avoid heavy top-level dependencies.
_MODEL_TYPE_MAPPING = (
    (
        "keras_hub.src.models.gemma4.gemma4_causal_lm",
        "Gemma4CausalLM",
        "gemma4",
    ),
    (
        "keras_hub.src.models.gemma3n.gemma3n_causal_lm",
        "Gemma3nCausalLM",
        "gemma3n",
    ),
    (
        "keras_hub.src.models.gemma3.gemma3_causal_lm",
        "Gemma3CausalLM",
        "gemma3",
    ),
    (
        "keras_hub.src.models.gemma.gemma_causal_lm",
        "GemmaCausalLM",
        "generic_model",
    ),
    (
        "keras_hub.src.models.qwen3_moe.qwen3_moe_causal_lm",
        "Qwen3MoeCausalLM",
        "qwen3",
    ),
    (
        "keras_hub.src.models.qwen3.qwen3_causal_lm",
        "Qwen3CausalLM",
        "qwen3",
    ),
    (
        "keras_hub.src.models.qwen_moe.qwen_moe_causal_lm",
        "QwenMoeCausalLM",
        "qwen2p5",
    ),
    (
        "keras_hub.src.models.qwen.qwen_causal_lm",
        "QwenCausalLM",
        "qwen2p5",
    ),
    # NOTE: LlmModelType does not have a dedicated "llama" field; map
    # Llama checkpoints to generic_model so the protobuf oneof stays valid.
    (
        "keras_hub.src.models.llama.llama_causal_lm",
        "LlamaCausalLM",
        "generic_model",
    ),
)


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


def _first_attr(obj, *names, default=None):
    """Return the first non-``None`` attribute from *obj*, or *default*."""
    if obj is None:
        return default
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


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


def _validate_export_args(
    model,
    path,
    tokenizer,
    backend_constraint,
    hf_tokenizer_path,
    prefill_seq_len,
):
    """Fail fast on invalid export arguments.

    Returns the normalized list of prefill sequence lengths. Importing
    ``litert_torch`` is deferred to the orchestrator so that the JAX
    ``jax_enable_x64`` side effect can be kept under one preserve/restore
    context that covers both import and tracing.
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
    # ``_get_cache_config`` runs.
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

    return prefill_seq_lens


def _build_prefill_inputs(
    prefill_seq_lens,
    num_layers,
    cache_length,
    num_kv_heads,
    head_dim,
    dtype,
    cache_layout,
    has_vision,
    has_audio,
    vision_cfg,
    audio_cfg,
    separate_vision_encoder,
    is_gemma4_vision,
    vision_output_dim,
):
    """Build a ``{seq_len: sample_inputs}`` map for every prefill bucket."""
    prefill_inputs_map = {}
    for seq_len in prefill_seq_lens:
        base = _build_sample_inputs(
            batch_size=1,
            seq_len=seq_len,
            num_layers=num_layers,
            cache_length=cache_length,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            cache_layout=cache_layout,
        )
        if has_vision:
            max_images = vision_cfg["max_images_per_prompt"]
            num_vision_tokens = vision_cfg["num_vision_tokens"]
            if separate_vision_encoder:
                tokens_per_image = (
                    num_vision_tokens // max_images if max_images else 1
                )
                base.update(
                    {
                        "mm_embedding": torch.zeros(
                            (
                                1 * max_images,
                                tokens_per_image,
                                vision_output_dim,
                            ),
                            dtype=dtype,
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
            elif is_gemma4_vision:
                base.update(
                    _build_gemma4_vision_sample_inputs(
                        batch_size=1,
                        max_images=max_images,
                        patch_size=vision_cfg.get("patch_size", 16),
                        image_size=vision_cfg["image_size"],
                        num_vision_tokens=num_vision_tokens,
                        seq_len=seq_len,
                        dtype=dtype,
                    )
                )
            else:
                base.update(
                    _build_vision_sample_inputs(
                        batch_size=1,
                        max_images=max_images,
                        image_size=vision_cfg["image_size"],
                        num_vision_tokens=num_vision_tokens,
                        seq_len=seq_len,
                        dtype=dtype,
                    )
                )
        if has_audio:
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
                    dtype=dtype,
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
    is_gemma4_vision=False,
):
    """Create concrete sample inputs for a separate vision-encoder signature."""
    device = "cpu"
    if is_gemma4_vision:
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
    prefill_adapter,
    decode_adapter,
    prefill_inputs_map,
    decode_inputs,
    prefill_seq_lens,
    quant_config,
    separate_vision_encoder,
    has_vision,
    is_gemma4_vision,
    vision_cfg,
    vision_output_dim,
    max_images,
    tokens_per_image,
    dtype,
    **kwargs,
):
    """Trace prefill/decode (and optional vision) signatures and convert."""
    # Defer torch-specific adapter imports until the backend has been verified
    # as torch, so that non-torch callers get the friendly backend error.
    from keras_hub.src.utils.litertlm.adapter import KerasHubVisionAdapter
    from keras_hub.src.utils.litertlm.adapter import (
        KerasHubVisionEncoderAdapter,
    )
    from keras_hub.src.utils.litertlm.adapter import _traceable_arange_scope
    from keras_hub.src.utils.litertlm.adapter import (
        _traceable_dot_product_attention_scope,
    )
    from keras_hub.src.utils.litertlm.adapter import _traceable_one_hot_scope
    from keras_hub.src.utils.litertlm.adapter import _traceable_repeat_scope
    from keras_hub.src.utils.litertlm.adapter import (
        _traceable_scatter_update_scope,
    )
    from keras_hub.src.utils.litertlm.adapter import _traceable_slice_scope
    from keras_hub.src.utils.litertlm.adapter import _traceable_take_scope

    with (
        _traceable_slice_scope(),
        _traceable_dot_product_attention_scope(),
        _traceable_one_hot_scope(),
        _traceable_repeat_scope(),
        _traceable_arange_scope(),
        _traceable_take_scope(),
        _traceable_scatter_update_scope(),
    ):
        vision_encoder_edge = None
        vision_adapter_edge = None

        # Optionally export the vision encoder and adapter as separate models.
        if separate_vision_encoder and has_vision:
            patch_size = vision_cfg.get("patch_size", 16)
            vision_encoder_inputs = _build_vision_encoder_sample_inputs(
                batch_size=1,
                max_images=max_images,
                image_size=vision_cfg["image_size"],
                patch_size=patch_size,
                dtype=dtype,
                is_gemma4_vision=is_gemma4_vision,
            )
            vision_adapter_inputs = _build_vision_adapter_sample_inputs(
                batch_size=1,
                max_images=max_images,
                tokens_per_image=tokens_per_image,
                vision_output_dim=vision_output_dim,
                dtype=dtype,
            )
            vision_encoder_adapter = KerasHubVisionEncoderAdapter(model).eval()
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

        # Chain one signature per prefill bucket plus the decode signature.
        signatures = []
        for seq_len in prefill_seq_lens:
            sig_name = (
                "prefill"
                if len(prefill_seq_lens) == 1
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

    return edge_model, vision_encoder_edge, vision_adapter_edge


def _assemble_bundle(
    path,
    temp_dir,
    model,
    tokenizer,
    backend_constraint,
    edge_model,
    separate_vision_encoder,
    has_vision,
    vision_encoder_edge,
    vision_adapter_edge,
    cache_length,
    vision_cfg,
    audio_cfg,
    hf_tokenizer_path,
):
    """Write TFLite files, bundle the tokenizer, and assemble ``.litertlm``."""
    if separate_vision_encoder and has_vision:
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
        model,
        tokenizer,
        cache_length,
        meta_path,
        vision_cfg=vision_cfg,
        audio_cfg=audio_cfg,
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
    if separate_vision_encoder and has_vision:
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
    if use_hf_tokenizer:
        builder.add_hf_tokenizer(tokenizer_path)
    else:
        builder.add_sentencepiece_tokenizer(tokenizer_path)
    builder.add_llm_metadata(meta_path)

    with open(path, "wb") as output_file:
        builder.build(output_file)

    return path


def export_to_litertlm(
    model,
    path,
    backend_constraint=None,
    prefill_seq_len=None,
    quant_config=None,
    separate_vision_encoder=False,
    hf_tokenizer_path=None,
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
    matches the upstream LiteRT-LM multimodal runtime contract.

    **Bucketing:** ``prefill_seq_len`` accepts either a single ``int`` or a
    ``list[int]``. When a list is provided (e.g.
    ``[32, 64, 128, 256, 512, 1024]``), the exporter traces one prefill
    signature per bucket. At runtime the LiteRT-LM executor dispatches to
    the smallest bucket that fits the actual prompt, avoiding wasted
    computation on padding. For multimodal models (e.g. Gemma3), bucketing
    is not supported because the vision attention mask computation requires
    cache length to equal input length.

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
            tracing the prefill signature(s). Defaults to the model's
            maximum sequence length. Each value must not exceed
            ``cache_length``.
        quant_config: Optional
            ``litert_torch.quantize.quant_config.QuantConfig`` for
            in-conversion quantization. Defaults to ``None`` (no
            quantization, FP32).
        separate_vision_encoder: bool. If ``True`` and the model has a vision
            encoder, export the vision encoder and a no-op vision adapter as
            separate ``VISION_ENCODER`` and ``VISION_ADAPTER`` TFLite models,
            and have ``PREFILL_DECODE`` consume pre-computed ``mm_embedding``
            tensors instead of raw images. Defaults to ``False``.
        hf_tokenizer_path: Optional str. Path to a HuggingFace
            ``tokenizer.json`` file to bundle instead of the model's native
            tokenizer. Use this for BytePair / HuggingFace tokenizers that
            cannot be materialized as a SentencePiece ``.spm`` file. When
            provided, the native tokenizer validation is skipped. If ``None``,
            SentencePiece tokenizers are bundled as ``.spm`` and any
            ``BytePairTokenizer`` subclass is automatically converted to
            ``tokenizer.json``. Defaults to ``None``.
        **kwargs: Additional kwargs forwarded to ``litert_torch`` signature
            tracing.

    Returns:
        The output ``path``.

    Raises:
        ValueError: If the backend is not ``"torch"``, if ``path`` does not
            end with ``.litertlm``, if the model lacks ``call_with_cache``,
            if ``backend_constraint`` is invalid, if any
            ``prefill_seq_len`` exceeds ``cache_length``, or if a multimodal
            model is exported with mismatched ``prefill_seq_len`` values.
        ImportError: If ``litert-torch`` or ``litert-lm-builder`` are not
            installed.
    """
    path = os.fspath(path)
    tokenizer = _get_tokenizer(model)
    prefill_seq_lens = _validate_export_args(
        model,
        path,
        tokenizer,
        backend_constraint,
        hf_tokenizer_path,
        prefill_seq_len,
    )

    # LiteRT-LM's JAX bridge defaults to the TPU platform if one is visible,
    # but export must run on CPU so it does not contend with other processes
    # using the TPU. We update the JAX config before any backend access; the
    # environment variable alone is not enough if JAX was imported before
    # ``keras_hub.src`` set it.
    try:
        import jax
    except ImportError:
        jax = None
    if jax is not None:
        jax.config.update("jax_platforms", "cpu")

    # Defer torch-specific adapter imports until after the backend check so
    # that a JAX/TF caller without torch gets the friendly backend error.
    from keras_hub.src.utils.litertlm.adapter import KerasHubLiteRTAdapter
    from keras_hub.src.utils.litertlm.adapter import _cpu_default_device_scope
    from keras_hub.src.utils.litertlm.adapter import _get_vision_encoder
    from keras_hub.src.utils.litertlm.adapter import _is_gemma4_vision_encoder

    cache_cfg = _get_cache_config(model)
    num_layers = cache_cfg["num_layers"]
    cache_length = cache_cfg["cache_length"]
    num_kv_heads = cache_cfg["num_kv_heads"]
    head_dim = cache_cfg["head_dim"]
    cache_layout = cache_cfg["cache_layout"]

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
    vision_cfg = _get_vision_config(model)
    audio_cfg = _get_audio_config(model)
    has_vision = vision_cfg is not None
    has_audio = audio_cfg is not None

    is_gemma4_vision = False
    vision_output_dim = None
    if has_vision:
        vision_encoder = _get_vision_encoder(model.backbone)
        is_gemma4_vision = _is_gemma4_vision_encoder(vision_encoder)
        vision_output_dim = getattr(vision_encoder, "output_dim", None)
        if vision_output_dim is None:
            # PaliGemma's ViT uses ``num_classes`` as the projected vision
            # dimension instead of ``output_dim``.
            vision_output_dim = getattr(vision_encoder, "num_classes", None)
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

    # Gemma3n runs vision/audio encoders inside the backbone and expects raw
    # pixel_values / input_features, so a separate vision encoder is not
    # meaningful for that architecture.
    if separate_vision_encoder and has_vision:
        call_params = set(inspect.signature(model.call_with_cache).parameters)
        if "pixel_values" in call_params:
            raise ValueError(
                "`separate_vision_encoder=True` is not supported for models "
                "that expect raw `pixel_values` (e.g. Gemma3n)."
            )

    # Multimodal models require cache_length == token_length due to how
    # Gemma3 computes bidirectional image attention masks. Enforce this.
    if has_vision and any(
        seq_len != cache_length for seq_len in prefill_seq_lens
    ):
        raise ValueError(
            f"Multimodal LiteRT-LM export currently requires all "
            f"`prefill_seq_len` values ({prefill_seq_lens}) to match the "
            f"cache_length ({cache_length}). This is a limitation of the "
            f"Gemma3 attention mask computation when cache length differs "
            f"from input length."
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
    with _cpu_default_device_scope():
        prefill_inputs_map = _build_prefill_inputs(
            prefill_seq_lens=prefill_seq_lens,
            num_layers=num_layers,
            cache_length=cache_length,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            cache_layout=cache_layout,
            has_vision=has_vision,
            has_audio=has_audio,
            vision_cfg=vision_cfg,
            audio_cfg=audio_cfg,
            separate_vision_encoder=separate_vision_encoder,
            is_gemma4_vision=is_gemma4_vision,
            vision_output_dim=vision_output_dim,
        )

        decode_inputs = _build_sample_inputs(
            batch_size=1,
            seq_len=1,
            num_layers=num_layers,
            cache_length=cache_length,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            cache_layout=cache_layout,
        )

        adapter = KerasHubLiteRTAdapter(
            model,
            num_layers,
            cache_length,
            separate_vision_encoder=(separate_vision_encoder and has_vision),
            cache_layout=cache_layout,
        )
        adapter.eval()

        prefill_adapter = _PrefillAdapter(adapter).eval()
        decode_adapter = _DecodeAdapter(adapter).eval()

        with _preserve_jax_x64_state():
            import litert_torch

            _validate_quant_config(quant_config)
            edge_model, vision_encoder_edge, vision_adapter_edge = (
                _trace_and_convert(
                    litert_torch,
                    model,
                    prefill_adapter,
                    decode_adapter,
                    prefill_inputs_map,
                    decode_inputs,
                    prefill_seq_lens,
                    quant_config,
                    separate_vision_encoder,
                    has_vision,
                    is_gemma4_vision,
                    vision_cfg,
                    vision_output_dim,
                    max_images,
                    tokens_per_image,
                    dtype,
                    **kwargs,
                )
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        _assemble_bundle(
            path,
            temp_dir,
            model,
            tokenizer,
            backend_constraint,
            edge_model,
            separate_vision_encoder,
            has_vision,
            vision_encoder_edge,
            vision_adapter_edge,
            cache_length,
            vision_cfg,
            audio_cfg,
            hf_tokenizer_path,
        )

    return path


def _get_cache_config(model):
    """Extract KV-cache dimensions and layout from the model."""
    backbone = model.backbone
    num_layers = _first_attr(backbone, "num_layers", "num_hidden_layers")
    if num_layers is None:
        raise ValueError(
            "Could not determine number of layers from model backbone. "
            "Expected `backbone.num_layers` or `backbone.num_hidden_layers`."
        )

    cache_length = _first_attr(backbone, "max_sequence_length")
    if cache_length is None:
        preprocessor = getattr(model, "preprocessor", None)
        cache_length = _first_attr(preprocessor, "sequence_length")
    if cache_length is None:
        raise ValueError(
            "Could not determine cache length from model backbone or "
            "preprocessor. Please specify `prefill_seq_len` or ensure the "
            "model has `max_sequence_length`."
        )

    num_kv_heads = _first_attr(
        backbone,
        "num_key_value_heads",
        "num_query_heads",
        "num_heads",
        "num_attention_heads",
        default=1,
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

    # Gemma3n uses a [B, L, 2, H, T, D] cache layout, whereas most other
    # KerasHub models use [B, L, 2, T, H, D].  We detect this from the
    # backbone class name so the adapter can build sample inputs with the
    # correct per-layer cache shape.
    cache_layout = "standard"
    if type(backbone).__name__.startswith("Gemma3n"):
        cache_layout = "gemma3n"

    return {
        "num_layers": num_layers,
        "cache_length": cache_length,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "cache_layout": cache_layout,
    }


def _get_vision_config(model):
    """Return vision metadata if *model* has a vision encoder, else ``None``."""
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
        # Gemma3n does not set backbone.image_size; read from the preprocessor
        # image converter first, then fall back to the encoder config.
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
            "`backbone.image_size`, `preprocessor.image_converter.image_size`, "
            "and `backbone.vision_encoder_config.image_shape[0]`."
        )
    # Image converters may report a (height, width) tuple; downstream code
    # currently assumes a square image, so use the height as the size.
    if isinstance(image_size, (list, tuple)):
        image_size = image_size[0]

    num_vision_tokens_per_image = getattr(
        backbone, "num_vision_tokens_per_image", None
    )
    if num_vision_tokens_per_image is None:
        # PaliGemma exposes the per-image token count via
        # ``image_sequence_length`` rather than ``num_vision_tokens_per_image``.
        num_vision_tokens_per_image = getattr(
            backbone, "image_sequence_length", None
        )
    if num_vision_tokens_per_image is None and preprocessor is not None:
        # Gemma3/Gemma3n expose the per-image token count on the preprocessor.
        num_vision_tokens_per_image = getattr(
            preprocessor, "num_vision_tokens_per_image", None
        )
    if num_vision_tokens_per_image is None:
        raise ValueError(
            "Could not determine `num_vision_tokens_per_image`. Searched "
            "`backbone.num_vision_tokens_per_image`, "
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


def _get_audio_config(model):
    """Return audio metadata if *model* has an audio encoder, else ``None``."""
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
    audio_input_feat_size = getattr(preprocessor, "audio_input_feat_size", None)
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


def _populate_vision_metadata(meta, model_type, vision_cfg, tokenizer):
    """Populate vision-related fields in the LlmMetadata protobuf."""
    image_size = vision_cfg["image_size"]
    patch_size = vision_cfg.get("patch_size")
    pool_size = vision_cfg.get("pool_size")

    if model_type in ("gemma3", "gemma3n"):
        subtype = getattr(meta.llm_model_type, model_type)
        subtype.start_of_image_token.token_str = _GEMMA3_START_OF_IMAGE_TOKEN
        subtype.end_of_image_token.token_str = _GEMMA3_END_OF_IMAGE_TOKEN
        subtype.image_tensor_height = image_size
        subtype.image_tensor_width = image_size
    elif model_type == "gemma4":
        subtype = meta.llm_model_type.gemma4
        subtype.start_of_image_token.token_str = _GEMMA4_START_OF_IMAGE_TOKEN
        subtype.end_of_image_token.token_str = _GEMMA4_END_OF_IMAGE_TOKEN
        if patch_size is not None:
            subtype.patch_width = patch_size
            subtype.patch_height = patch_size
            subtype.max_num_patches = (image_size // patch_size) ** 2
        if pool_size is not None:
            subtype.pooling_kernel_size = pool_size


def _populate_audio_metadata(meta, model_type, audio_cfg, tokenizer):
    """Populate audio-related fields in the LlmMetadata protobuf."""
    if model_type in ("gemma3n", "gemma4"):
        subtype = getattr(meta.llm_model_type, model_type)
        subtype.start_of_audio_token.token_str = _AUDIO_START_TOKEN
        subtype.end_of_audio_token.token_str = _AUDIO_END_TOKEN


def _build_llm_metadata(
    model, tokenizer, max_num_tokens, path, vision_cfg=None, audio_cfg=None
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

    end_id = getattr(tokenizer, "end_token_id", None)
    if end_id is not None:
        meta.stop_tokens.add().token_ids.ids.append(int(end_id))

    # ``<end_of_turn>`` is an optional stop token for some Gemma/SentencePiece
    # tokenizers. Only look it up when the tokenizer exposes ``token_to_id``,
    # and swallow the specific lookup-failure exceptions so a missing special
    # token does not abort export.
    if hasattr(tokenizer, "token_to_id"):
        try:
            eot_id = tokenizer.token_to_id("<end_of_turn>")
        except (KeyError, ValueError):
            eot_id = None
        if eot_id is not None:
            unk_id = getattr(tokenizer, "_unk_token_id", None)
            if eot_id != unk_id:
                meta.stop_tokens.add().token_ids.ids.append(int(eot_id))

    meta.max_num_tokens = int(max_num_tokens)

    model_type = _detect_llm_model_type(model)
    getattr(meta.llm_model_type, model_type).SetInParent()

    # Populate vision fields for supported model types.
    if vision_cfg is not None:
        _populate_vision_metadata(meta, model_type, vision_cfg, tokenizer)

    # Populate audio fields for supported model types.
    if audio_cfg is not None:
        _populate_audio_metadata(meta, model_type, audio_cfg, tokenizer)

    with open(path, "wb") as f:
        f.write(meta.SerializeToString())


def _detect_llm_model_type(model):
    """Return the LiteRT-LM LlmModelType name for *model*.

    Uses ``isinstance`` checks against the module-level
    ``_MODEL_TYPE_MAPPING`` to avoid mis-identifying user-defined
    subclasses. Unrecognized models map to ``"generic_model"``.
    """
    for module_path, class_name, model_type in _MODEL_TYPE_MAPPING:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            if isinstance(model, cls):
                return model_type
        except ImportError:
            pass

    return "generic_model"


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
