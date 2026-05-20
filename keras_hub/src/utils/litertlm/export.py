"""Export KerasHub CausalLM models to LiteRT-LM `.litertlm` bundles."""

import os
import tempfile

import keras
import torch

try:
    import litert_torch
except ImportError:
    litert_torch = None

from keras_hub.src.utils.litertlm.adapter import KerasHubLiteRTAdapter
from keras_hub.src.utils.litertlm.adapter import _traceable_slice_update_scope
from keras_hub.src.utils.preset_utils import TOKENIZER_ASSET_DIR


def export_to_litertlm(
    model,
    path,
    backend_constraint=None,
    prefill_seq_len=None,
    **kwargs,
):
    """Export a KerasHub CausalLM model to a LiteRT-LM bundle.

    This exports the model with ``prefill`` and ``decode`` signatures
    required by the LiteRT-LM executor, bundles the SentencePiece tokenizer,
    and writes an ``LlmMetadata`` protobuf into the ``.litertlm`` artifact.

    Args:
        model: A KerasHub ``CausalLM`` instance with an attached preprocessor
            and tokenizer.
        path: str. Path to save the ``.litertlm`` file.
        backend_constraint: Optional LiteRT-LM backend constraint, such as
            ``"cpu"`` or ``"gpu"``.
        prefill_seq_len: int. Sequence length used when tracing the prefill
            signature.  Defaults to the model's maximum sequence length.
            *Note:* dynamic sequence length is not yet supported, so prompts
            should be padded/truncated to this exact length at runtime.
        **kwargs: Additional kwargs forwarded to ``litert_torch`` conversion.

    Returns:
        The output ``path``.
    """
    if keras.config.backend() != "torch":
        raise ValueError(
            "LiteRT-LM export is only supported with the PyTorch backend. "
            f"Current backend: {keras.config.backend()}."
        )

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
        valid_backends = {"cpu", "gpu", "npu", "gpu_artisan"}
        if backend_constraint.lower() not in valid_backends:
            raise ValueError(
                f"Invalid backend_constraint: {backend_constraint!r}. "
                f"Must be one of {sorted(valid_backends)}."
            )

    tokenizer = _get_tokenizer(model)
    _validate_sentencepiece_tokenizer(tokenizer)
    cache_cfg = _get_cache_config(model)
    num_layers = cache_cfg["num_layers"]
    cache_length = cache_cfg["cache_length"]
    num_kv_heads = cache_cfg["num_kv_heads"]
    head_dim = cache_cfg["head_dim"]

    if prefill_seq_len is None:
        prefill_seq_len = cache_length

    if prefill_seq_len > cache_length:
        raise ValueError(
            f"prefill_seq_len ({prefill_seq_len}) cannot exceed "
            f"cache_length ({cache_length})."
        )

    # Build sample inputs for tracing.
    dtype = _torch_dtype_from_model(model)
    prefill_inputs = _build_sample_inputs(
        batch_size=1,
        seq_len=prefill_seq_len,
        num_layers=num_layers,
        cache_length=cache_length,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    decode_inputs = _build_sample_inputs(
        batch_size=1,
        seq_len=1,
        num_layers=num_layers,
        cache_length=cache_length,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    adapter = KerasHubLiteRTAdapter(model, num_layers, cache_length)
    adapter.eval()

    # Prefill and decode wrappers give litert_torch clean module boundaries.
    class _PrefillAdapter(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, tokens, input_pos, mask=None, **kv_cache):
            return self.base.forward_prefill(
                tokens, input_pos, mask, **kv_cache
            )

    class _DecodeAdapter(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, tokens, input_pos, mask=None, **kv_cache):
            return self.base.forward_decode(tokens, input_pos, mask, **kv_cache)

    prefill_adapter = _PrefillAdapter(adapter).eval()
    decode_adapter = _DecodeAdapter(adapter).eval()

    if litert_torch is None:
        raise ImportError(
            "LiteRT-LM export requires `litert-torch`. "
            "Install it with: pip install litert-torch"
        )

    with _traceable_slice_update_scope():
        edge_model = (
            litert_torch.signature(
                "prefill",
                prefill_adapter,
                sample_kwargs=prefill_inputs,
                **kwargs,
            )
            .signature(
                "decode",
                decode_adapter,
                sample_kwargs=decode_inputs,
                **kwargs,
            )
            .convert()
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        tflite_path = os.path.join(temp_dir, "model.tflite")
        edge_model.export(tflite_path)

        tokenizer_path = _materialize_sentencepiece_tokenizer(
            tokenizer, temp_dir
        )

        meta_path = os.path.join(temp_dir, "llm_metadata.pb")
        _build_llm_metadata(model, cache_length, meta_path)

        # Package the TFLite model, tokenizer, and metadata into a
        # `.litertlm` bundle using the public LiteRT-LM builder API.
        litert_lm_builder = _import_litert_lm_builder()
        builder = litert_lm_builder.LitertLmFileBuilder()
        builder.add_system_metadata(
            litert_lm_builder.Metadata(
                key="Authors",
                value="KerasHub",
                dtype=litert_lm_builder.DType.STRING,
            )
        )
        # `add_tflite_model` is the official builder API for adding the
        # inference graph to the `.litertlm` bundle.  The bundle may
        # contain multiple TFLite models (e.g. prefill/decode, embedder,
        # vision encoder) identified by `TfLiteModelType`.
        builder.add_tflite_model(
            tflite_path,
            litert_lm_builder.TfLiteModelType.PREFILL_DECODE,
            backend_constraint=backend_constraint,
        )
        builder.add_sentencepiece_tokenizer(tokenizer_path)
        builder.add_llm_metadata(meta_path)

        with open(path, "wb") as output_file:
            builder.build(output_file)

    return path


def _get_cache_config(model):
    """Extract KV-cache dimensions from the model."""
    backbone = model.backbone
    num_layers = backbone.num_layers

    cache_length = getattr(backbone, "max_sequence_length", None)
    if cache_length is None:
        preprocessor = getattr(model, "preprocessor", None)
        if preprocessor is not None:
            cache_length = getattr(preprocessor, "sequence_length", None)
    if cache_length is None:
        raise ValueError(
            "Could not determine cache length from model backbone or "
            "preprocessor. Please specify `prefill_seq_len` or ensure the "
            "model has `max_sequence_length`."
        )

    num_kv_heads = getattr(backbone, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(
            backbone, "num_heads", getattr(backbone, "num_query_heads", 1)
        )

    head_dim = getattr(backbone, "head_dim", None)
    if head_dim is None:
        hidden_dim = getattr(backbone, "hidden_dim", None)
        num_qh = getattr(
            backbone, "num_query_heads", getattr(backbone, "num_heads", None)
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
    }


def _build_sample_inputs(
    batch_size,
    seq_len,
    num_layers,
    cache_length,
    num_kv_heads,
    head_dim,
    dtype=torch.float32,
):
    """Create concrete sample tensors for one signature."""
    device = "cpu"
    tokens = torch.zeros(
        (batch_size, seq_len), dtype=torch.int32, device=device
    )
    input_pos = torch.arange(seq_len, dtype=torch.int32, device=device)
    if seq_len == 1:
        # Decode: input_pos is the current position (e.g. [0]).
        input_pos = torch.zeros((1,), dtype=torch.int32, device=device)
    # Causal mask – all ones.  Most KerasHub models ignore this in
    # ``call_with_cache``, but LiteRT-LM executors may expect it.
    mask = torch.ones(
        (batch_size, 1, seq_len, cache_length),
        dtype=dtype,
        device=device,
    )
    kv_cache = {}
    for i in range(num_layers):
        shape = (batch_size, cache_length, num_kv_heads, head_dim)
        kv_cache[f"kv_cache_k_{i}"] = torch.zeros(
            shape, dtype=dtype, device=device
        )
        kv_cache[f"kv_cache_v_{i}"] = torch.zeros(
            shape, dtype=dtype, device=device
        )

    sample = {
        "tokens": tokens,
        "input_pos": input_pos,
        "mask": mask,
    }
    sample.update(kv_cache)
    return sample


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


def _build_llm_metadata(model, max_num_tokens, path):
    """Serialize an ``LlmMetadata`` protobuf to *path*."""
    from litert_lm_builder.litertlm_builder import llm_metadata_pb2

    meta = llm_metadata_pb2.LlmMetadata()

    # Start / stop tokens
    tokenizer = _get_tokenizer(model)
    start_id = getattr(tokenizer, "start_token_id", None)
    if start_id is not None:
        meta.start_token.token_ids.ids.append(int(start_id))

    end_id = getattr(tokenizer, "end_token_id", None)
    if end_id is not None:
        meta.stop_tokens.add().token_ids.ids.append(int(end_id))

    # Gemma3 (and similar chat-tuned models) use <end_of_turn> as the turn
    # separator.  The LiteRT-LM runtime needs to know about it so generation
    # stops after the assistant finishes its turn.
    try:
        eot_id = tokenizer.token_to_id("<end_of_turn>")
        if eot_id is not None:
            meta.stop_tokens.add().token_ids.ids.append(int(eot_id))
    except Exception:
        pass

    meta.max_num_tokens = int(max_num_tokens)

    # Use model-specific type when known so the LiteRT-LM runtime applies
    # the correct chat template and stop-token behaviour.
    model_type = _detect_llm_model_type(model)
    getattr(meta.llm_model_type, model_type).SetInParent()

    with open(path, "wb") as f:
        f.write(meta.SerializeToString())


def _detect_llm_model_type(model):
    """Return the LiteRT-LM LlmModelType name for *model*."""
    cls_name = type(model).__name__
    if "Gemma3" in cls_name:
        return "gemma3"
    if "Gemma" in cls_name:
        return "generic_model"  # older Gemma variants
    if "Qwen3" in cls_name:
        return "qwen3"
    if "Qwen2" in cls_name:
        return "qwen2p5"
    return "generic_model"


def _torch_dtype_from_model(model):
    """Return a ``torch.dtype`` matching the model's compute dtype."""
    compute_dtype = getattr(model, "compute_dtype", None)
    if compute_dtype is None:
        compute_dtype = getattr(model.backbone, "compute_dtype", "float32")
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
