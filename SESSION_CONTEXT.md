# LiteRT-LM Export PR Session Context

**Date:** 2026-06-19
**Branch:** `torch-backend-litert-minimal-litertlm`
**Fork remote:** `pctablet505/keras-hub`
**Upstream PR:** https://github.com/keras-team/keras-hub/pull/2705
**Latest commit:** `4249813` (Iteration 14)

## Goal
Add a PyTorch-backend LiteRT-LM export path for KerasHub CausalLM models,
with dedicated tests for all supported model families.

## Current Status
- All supported CausalLM families have `test_litertlm_export` in their model
  test files.
- CPU-only export tracing is enforced via `_cpu_default_device_scope()` and
  restored after export.
- Runtime generation smoke test is wired into `test_export_tiny_gemma`.
- Design doc (PR body) contains supported/unsupported model table and
  architecture notes.

## Supported Models (SentencePiece CausalLM)
- Gemma, Gemma3, Gemma3n, Gemma4
- Llama, Mistral, Mixtral
- PaliGemma, Phi3

## Unsupported Models
- BytePair tokenizer families: Bloom, Falcon, GPT2, GPTNeoX, GPT-OSS,
  Llama3, OPT, Qwen variants, SmolLM3
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
- `export_test.py`: 12 passed, 3 skipped, 4 subtests passed (57.53s)
- Full targeted suite (iteration 15): 32 passed, 17 skipped, 4 subtests passed

## Open Items / Known Gaps
- Audio export for Gemma3n is not enabled end-to-end (audio encoder exists but
  is not wired into the export sample inputs path for separate encoders).
- Quantization configs are accepted but not covered by dedicated tests.
- Externalized weights / streaming export is not supported.
- Generation smoke test output with random weights is garbage (`<pad>` tokens);
  it only verifies the runtime executes without error.

## Key Files Changed
- `keras_hub/src/utils/litertlm/export.py`
- `keras_hub/src/utils/litertlm/adapter.py`
- `keras_hub/src/utils/litertlm/export_test.py`
- `keras_hub/src/tests/test_case.py`
- `keras_hub/src/tests/mocks/mock_gemma3_tokenizer.py`
- Model test files for the 9 supported families

## Environment
- Backend: PyTorch (`KERAS_BACKEND=torch`)
- Required packages: `litert-torch`, `litert-lm-builder`, `litert-lm`
