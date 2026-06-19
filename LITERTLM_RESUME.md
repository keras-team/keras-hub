# LiteRT-LM Export Work — Resume Notes

**Branch:** `pctablet505/torch-backend-litert-minimal-litertlm`  
**PR:** https://github.com/keras-team/keras-hub/pull/2705  
**Last updated:** 2026-06-18

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
    - Gemma, Gemma3, Gemma4, Mistral, Mixtral, Phi3, Llama, PaliGemma, Gemma3n
    - Gemma3n text-only and baked-in vision tests pass
    - Gemma3n separate-vision-encoder test is skipped (MobileNetV5 projection is inside the backbone)
13. **Torch-export-friendly `one_hot` patch** to avoid the unlowerable `aten._assert_async.msg` op introduced by `torch.nn.functional.one_hot` in torch >= 2.12.

## Key Files

| File | Purpose |
|------|---------|
| `keras_hub/src/utils/litertlm/export.py` | Main export pipeline |
| `keras_hub/src/utils/litertlm/adapter.py` | PyTorch adapter, separate vision encoders/adapters |
| `keras_hub/src/tests/test_case.py` | `run_litertlm_export_test` helper |
| `keras_hub/src/utils/litertlm/export_test.py` | Core tests (Gemma, Gemma3, bucketing, parity) |
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
- `litert-torch==0.9.1`
- `litert-lm==0.10.0`
- `litert-lm-builder==0.13.0`

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

## Known Issues / Blockers

1. **JAX CI failures** on `keras-stable` are unrelated to this PR. Failures are in `samplers/*_sampler_test.py` and `utils/transformers/export/gemma*_test.py` due to int64/int32 mismatch in `dynamic_update_slice` / `dynamic_slice`.
2. **Gemma3n separate vision encoder** not supported yet — MobileNetV5 does not expose a single projected vision dimension; Gemma3n applies reshape / sqrt-scaling / `embed_vision` inside the backbone after the encoder. Baked-in (single-model) Gemma3n vision export works.
3. **Audio encoder separation** blocked upstream by `litert-torch` issue #1039.
4. **BytePair / HuggingFace tokenizers** intentionally not supported. The LiteRT-LM / MediaPipe LLM Inference runtime contract only supports SentencePiece model protobuf files as the tokenizer model. BPE/HF tokenizers would need a lossy conversion to SentencePiece (the upstream `litert_torch/generative/tools/tokenizer_to_sentencepiece.py` notes ~1% token-ID mismatch for Llama3.2), which is out of scope for this PR.

## Pre-Commit Status

Passed on changed files:
```bash
pre-commit run --files keras_hub/src/utils/litertlm/adapter.py keras_hub/src/utils/litertlm/export.py keras_hub/src/utils/litertlm/*_litertlm_export_test.py
```

## Next Steps / Open Work

- [x] Run full `pytest keras_hub/src/utils/litertlm/ -v` suite when time allows.
- [ ] Run `pre-commit run --all-files` before marking PR ready for review.
- [x] Investigate Gemma3n KV-cache layout and 4-D vision encoder support.
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
