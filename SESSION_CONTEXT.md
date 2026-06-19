# LiteRT-LM Export PR Session Context

**Date:** 2026-06-19
**Branch:** `torch-backend-litert-minimal-litertlm`
**Fork remote:** `pctablet505/keras-hub`
**Upstream PR:** https://github.com/keras-team/keras-hub/pull/2705
**Latest commit:** `c1b17ed` (HF tokenizer support)

## Goal
Add a PyTorch-backend LiteRT-LM export path for KerasHub CausalLM models,
with dedicated tests for all supported model families, including BytePair
families via HuggingFace tokenizer conversion.

## Current Status
- All SentencePiece-supported CausalLM families have `test_litertlm_export` in
  their model test files.
- CPU-only export tracing is enforced via `_cpu_default_device_scope()` and
  restored after export.
- Runtime generation smoke test is wired into `test_export_tiny_gemma`.
- Design doc (PR body) contains supported/unsupported model table and
  architecture notes.
- HuggingFace tokenizer support added:
  - `hf_tokenizer_path` user override.
  - Auto-conversion for GPT2, Llama3, Qwen3 BytePair tokenizers.

## Supported Models

### SentencePiece
- Gemma, Gemma3, Gemma3n, Gemma4
- Llama, Mistral, Mixtral
- PaliGemma, Phi3

### Auto-converted BytePair → HF tokenizer.json
- GPT2, Llama3, Qwen3

### User-provided HF tokenizer.json
- Any BytePair family (Bloom, Falcon, GPTNeoX, GPT-OSS, OPT, Qwen variants,
  SmolLM3, etc.) via `hf_tokenizer_path`.

## Unsupported Models
- Non-transformer: RWKV7
- Non-generative: PARSeq

## Test Commands
```bash
# Full targeted suite (use -n 4 if -n auto flakes)
KERAS_BACKEND=torch pytest keras_hub/src/utils/litertlm/ \
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

## Last Test Result
- `keras_hub/src/utils/litertlm/` full suite: **28 passed, 18 skipped, 8 subtests passed in 76.93s**
- Per-model suite (all 24 families): **24 passed in 102.92s**

## Pixel 9 Verification
- `tiny_gemma3_bucketed.litertlm` (~500 KB): **PASSED**
- `gemma3_270m_it.litertlm` (~1 GB): **PASSED**
- `gemma3_270m_it_int4_bucketed_32_64_128.litertlm`: **FAILED** — `GATHER_ND` op fails on device CPU delegate (INT4 compatibility issue, not exporter issue).
- HF-converted tiny GPT2 bundle (`tiny_gpt2_hf_char.litertlm`): **PASSED** — generated `hohohohohohohohohohohohohoho` from prompt `hi`.

## Open Items / Known Gaps
- Audio export for Gemma3n is not enabled end-to-end (audio encoder exists but
  is not wired into the export sample inputs path for separate encoders).
- Quantization configs are accepted but not covered by dedicated tests.
- Externalized weights / streaming export is not supported.
- Generation smoke test output with random weights is garbage; it only verifies
  the runtime executes without error.
- INT4 weight-only models fail on Pixel 9 CPU delegate (`GATHER_ND` invoke
  error).

## Key Files Changed
- `keras_hub/src/utils/litertlm/export.py`
- `keras_hub/src/utils/litertlm/adapter.py`
- `keras_hub/src/utils/litertlm/hf_tokenizer_converter.py` (new)
- `keras_hub/src/utils/litertlm/export_test.py`
- `keras_hub/src/tests/test_case.py`
- `keras_hub/src/tests/mocks/mock_gemma3_tokenizer.py`
- Model test files for the 9 supported SentencePiece families

## Environment
- Backend: PyTorch (`KERAS_BACKEND=torch`)
- Required packages: `litert-torch`, `litert-lm-builder`, `litert-lm`
