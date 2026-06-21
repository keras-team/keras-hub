# LiteRT-LM Export Work — Resume Notes

**Branch:** `pctablet505/torch-backend-litert-minimal-litertlm`  
**PR:** https://github.com/keras-team/keras-hub/pull/2705  
**Last updated:** 2026-06-20

## Quick Start

```bash
cd /home/pctablet505/Projects/keras-hub
source /home/pctablet505/Projects/gemmademo-litert-export/.venv/bin/activate

# Run a single family test
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=torch pytest keras_hub/src/utils/litertlm/pali_gemma_litertlm_export_test.py -v

# Run all litertlm tests (time-consuming)
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=torch pytest keras_hub/src/utils/litertlm/ -v

# Run with parallelism (use with care; exports are CPU/RAM heavy)
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=torch pytest keras_hub/src/utils/litertlm/ -v -n auto

# Pre-commit on changed files
pre-commit run --files keras_hub/src/utils/litertlm/adapter.py keras_hub/src/utils/litertlm/export.py keras_hub/src/utils/litertlm/*_litertlm_export_test.py
```

## What Has Been Implemented

1. **`CausalLM.export(..., format="litertlm")` dispatch** in `keras_hub/src/models/causal_lm.py`.
2. **PyTorch-only backend guard** — exporter raises if backend is not torch.
3. **KV-cache dtype** derived from model compute dtype, not hardcoded FP32.
4. **Prefill + decode signatures** traced via `litert_torch`.
5. **Bucketing** for text-only models (`prefill_seq_len=[32, 64, 128]`).
6. **SentencePiece tokenizer** materialization and bundling.
7. **`LlmMetadata` construction** with correct `LlmModelType` mapping.
   - Llama maps to `generic_model` because the protobuf lacks a `llama` field.
8. **Multimodal sample inputs** for Gemma3 (raw images), Gemma4 (patches), and audio mel.
9. **Separate vision encoder/adapter export** (`separate_vision_encoder=True`) for Gemma3, Gemma4, and PaliGemma.
10. **PaliGemma support** — handles `vit_encoder` and single-image `[B, H, W, 3]` encoders.
11. **`run_litertlm_export_test` test helper** in `keras_hub/src/tests/test_case.py` with:
    - Per-step timing output
    - TFLite signature verification
    - Optional end-to-end generation smoke test via `litert_lm.Engine`
12. **Broad test matrix** for tiny random-weight models:
    - SentencePiece families: Gemma, Gemma3, Gemma4, Mistral, Mixtral, Phi3, Llama, PaliGemma, Gemma3n
    - HF-converted BytePair families: GPT2, Llama3, Qwen3
    - Gemma3n text-only and baked-in vision tests pass
    - Gemma3n separate-vision-encoder test is skipped (MobileNetV5 projection is inside the backbone)
13. **Torch-export-friendly `one_hot` patch** to avoid the unlowerable `aten._assert_async.msg` op introduced by `torch.nn.functional.one_hot` in torch >= 2.12.
14. **HuggingFace tokenizer support**:
    - `hf_tokenizer_path` argument for user-provided `tokenizer.json`
    - Auto-conversion of KerasHub `BytePairTokenizer` to HF `tokenizer.json` for GPT2, Llama3, Qwen3
    - New converter module: `keras_hub/src/utils/litertlm/hf_tokenizer_converter.py`

## Key Files

| File | Purpose |
|------|---------|
| `keras_hub/src/utils/litertlm/export.py` | Main export pipeline |
| `keras_hub/src/utils/litertlm/adapter.py` | PyTorch adapter, separate vision encoders/adapters |
| `keras_hub/src/utils/litertlm/hf_tokenizer_converter.py` | BytePair → HF `tokenizer.json` converter |
| `keras_hub/src/tests/test_case.py` | `run_litertlm_export_test` helper |
| `keras_hub/src/utils/litertlm/export_test.py` | Core tests (Gemma, Gemma3, bucketing, parity, HF tokenizer) |
| `keras_hub/src/utils/litertlm/gemma4_litertlm_export_test.py` | Gemma4 tests |
| `keras_hub/src/utils/litertlm/pali_gemma_litertlm_export_test.py` | PaliGemma tests |
| `keras_hub/src/utils/litertlm/llama_litertlm_export_test.py` | Llama tests |
| `keras_hub/src/utils/litertlm/mistral_litertlm_export_test.py` | Mistral tests |
| `keras_hub/src/utils/litertlm/mixtral_litertlm_export_test.py` | Mixtral tests |
| `keras_hub/src/utils/litertlm/phi3_litertlm_export_test.py` | Phi3 tests |
| `keras_hub/src/utils/litertlm/gemma3n_litertlm_export_test.py` | Gemma3n tests (text-only baked-in passes; separate vision encoder skipped) |

## Environment

```bash
source /home/pctablet505/Projects/gemmademo-litert-export/.venv/bin/activate
```

Key packages:
- `keras` from GitHub master
- `keras-hub` from local repo (editable)
- `litert-torch==0.10.0`
- `litert-lm==0.10.0`
- `litert-lm-builder==0.13.0`

System notes:
- Run tests with `CUDA_VISIBLE_DEVICES=""` to keep the `litert_torch` JAX bridge on CPU and avoid device-mismatch / aliasing issues.
- `libvulkan1` is required so `litert_lm.Engine` can load the CPU/GPU delegate in runtime smoke tests (`sudo apt install libvulkan1`).

## Verified End-to-End Generation in Python

```python
import litert_lm
engine = litert_lm.Engine(
    "/home/pctablet505/Projects/gemmademo-litert-export/gemma3_270m_it_int4_bucketed_32_64_128.litertlm",
    max_num_tokens=32,
)
conversation = engine.create_conversation()
print(conversation.send_message("What is the capital of France?"))
```

Works and returns text. With dummy-weight tiny models the output is meaningless but proves the bundle loads.

## Recent Root-Cause Fixes

1. **`litert_torch` constant-deduplication bug (scoped keras-hub workaround)**
   - `litert_torch.backend.inline_consts._tensor_fingerprint` caches constants by `(device, shape, stride, untyped_storage().data_ptr())`. The `data_ptr()` is an ephemeral address that can be reused for distinct constants and can alias key/value projection weights that share backing storage.
   - `adapter.py` provides a scoped context manager (`_litert_constant_fingerprint_scope`) that patches `_tensor_fingerprint` to use `id(tensor.untyped_storage())` plus `tensor.storage_offset()`. The patch is applied only during `converter.convert()`, avoiding import-time global side effects.

2. **TFLite output-buffer aliasing for KV caches**
   - The exported decode step was returning KV-cache tensors that TFLite could alias with intermediate activation buffers, causing sporadic corruption of cached key/value values (observed as large mismatches in the multimodal parity test).
   - `_call_with_cache` now clones the stacked updated cache before unstacking, and `_unstack_kv_cache` clones each per-layer slice, so the returned KV-cache outputs are independent buffers.
   - The `slice_update` patch was switched from `index_copy_` (in-place scatter on a cloned base) to a `torch.where`-based scatter into a full-shaped zero buffer, avoiding in-place mutations that the TFLite runtime may fuse into aliased buffers.

3. **Environment-sensitive device mismatches**
   - Tests must be run with `CUDA_VISIBLE_DEVICES=""`. When a non-functional CUDA device is visible, the `litert_torch` JAX bridge can place tracers on `cuda:0` while PyTorch sample inputs stay on CPU, causing `Unhandled FakeTensor Device Propagation` errors and nondeterministic numeric mismatches.

## Addressed PR Review Feedback

1. **Hardcoded Float32 KV caches** — KV-cache sample tensors and TFLite I/O specs already derive their dtype from `model.compute_dtype` via `_torch_dtype_from_model`; default `float32` arguments are only fallback values for the helper builders.
2. **Internal LiteRT API dependency** — The `llm_metadata_pb2` import is now wrapped in a `try/except` with a descriptive error message and an explanatory code comment.
3. **Backend guardrails** — `export_to_litertlm` raises `ValueError` immediately when `keras.config.backend() != "torch"`.
4. **Global side effects from patching** — The `_tensor_fingerprint` patch is no longer applied at import time; it is scoped to the `converter.convert()` call via `_litert_constant_fingerprint_scope`. Existing `slice_update`, `one_hot`, and SDPA replacements already use `unittest.mock.patch.object` context managers.

## Known Issues / Blockers

1. **JAX CI failures** on `keras-stable` are unrelated to this PR. Failures are in `samplers/*_sampler_test.py` and `utils/transformers/export/gemma*_test.py` due to int64/int32 mismatch in `dynamic_update_slice` / `dynamic_slice`.
2. **Gemma3n separate vision encoder** not supported yet — MobileNetV5 does not expose a single projected vision dimension; Gemma3n applies reshape / sqrt-scaling / `embed_vision` inside the backbone after the encoder. Baked-in (single-model) Gemma3n vision export works.
3. **Audio encoder separation** blocked upstream by `litert-torch` issue #1039.
4. **BytePair / HuggingFace tokenizers** intentionally not supported as first-class LiteRT-LM tokenizer models. The runtime contract primarily supports SentencePiece protobuf files. BPE/HF tokenizers would need a lossy conversion to SentencePiece (the upstream `litert_torch/generative/tools/tokenizer_to_sentencepiece.py` notes ~1% token-ID mismatch for Llama3.2), which is out of scope for this PR. The exporter still bundles auto-converted HF `tokenizer.json` files for GPT2/Llama3/Qwen3 when a BPE tokenizer is provided.

## Latest Test Result

Full targeted LiteRT-LM suite (with `-n auto`):

```bash
CUDA_VISIBLE_DEVICES="" KERAS_BACKEND=torch pytest \
  keras_hub/src/utils/litertlm/ \
  keras_hub/src/models/gemma/gemma_causal_lm_test.py::GemmaCausalLMTest::test_litertlm_export \
  keras_hub/src/models/gemma3/gemma3_causal_lm_test.py::Gemma3CausalLMTest::test_litertlm_export \
  keras_hub/src/models/gemma3n/gemma3n_causal_lm_test.py::Gemma3nCausalLMTest::test_litertlm_export \
  keras_hub/src/models/gemma4/gemma4_causal_lm_test.py::Gemma4CausalLMTest::test_litertlm_export \
  keras_hub/src/models/llama/llama_causal_lm_test.py::LlamaCausalLMTest::test_litertlm_export \
  keras_hub/src/models/mistral/mistral_causal_lm_test.py::MistralCausalLMTest::test_litertlm_export \
  keras_hub/src/models/mixtral/mixtral_causal_lm_test.py::MixtralCausalLMTest::test_litertlm_export \
  keras_hub/src/models/pali_gemma/pali_gemma_causal_lm_test.py::PaliGemmaCausalLMTest::test_litertlm_export \
  keras_hub/src/models/phi3/phi3_causal_lm_test.py::Phi3CausalLMTest::test_litertlm_export \
  -n auto -q
```

Result (2026-06-20, after review fixes): **28 passed, 18 skipped, 8 subtests passed in 238.82s** (all `keras_hub/src/utils/litertlm/` tests).

Per-model `test_litertlm_export` suite (supported families): **12 passed in 173.93s** across Gemma, Gemma3, Gemma3n, Gemma4, Llama, Mistral, Mixtral, PaliGemma, Phi3, Qwen3, Llama3, GPT2.

## Pixel 9 Verification
- `tiny_gemma3_bucketed.litertlm` (~500 KB): instrumented test **PASSED**
- `gemma3_270m_it.litertlm` (~1 GB): instrumented test **PASSED**
- HF-converted tiny GPT2 bundle: instrumented test **PASSED** (prompt `hi` → generated `hohohohohohohohohohohohohoho`)
- `gemma3_270m_it_int4_bucketed_32_64_128.litertlm`: **FAILED** with `GATHER_ND failed to invoke` on device CPU delegate (INT4 runtime issue, not exporter).

## Pre-Commit Status

Passed on changed files:
```bash
pre-commit run --files keras_hub/src/utils/litertlm/adapter.py keras_hub/src/utils/litertlm/export.py keras_hub/src/utils/litertlm/*_litertlm_export_test.py
```

## Next Steps / Open Work

- [x] Run full `pytest keras_hub/src/utils/litertlm/ -v` suite.
- [ ] Run `pre-commit run --all-files` before marking PR ready for review.
- [x] Investigate Gemma3n KV-cache layout and 4-D vision encoder support.
- [x] Fix core `export_test.py` numeric parity failures (constant dedup + buffer aliasing).
- [ ] Add BytePair / HuggingFace tokenizer support (future).
- [ ] Add audio encoder separation once upstream #1039 is resolved.
- [ ] Consider adding quantized-export tests (INT4 weight-only) to the matrix.
- [x] Update PR description design doc from `LITERTLM_RESUME.md` as work progresses.

## Useful Commands

```bash
# Pull latest branch on a new machine
git fetch pctablet505
git checkout torch-backend-litert-minimal-litertlm
git reset --hard pctablet505/torch-backend-litert-minimal-litertlm

# View PR status
gh pr view 2705 --repo keras-team/keras-hub

# View recent commits
git log --oneline -15
```
