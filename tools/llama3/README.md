# Llama3 Export and Verification Tools

This directory contains tools for exporting Llama3 models from KerasHub to
HuggingFace Transformers format and verifying that exported models produce
numerically equivalent outputs.

## Files

| File | Description |
|------|-------------|
| `export_llama3_to_hf.py` | Convert a KerasHub Llama3 preset to HuggingFace safetensors format |
| `verify_llama3_export.py` | Verify a HuggingFace export against the original KerasHub model |
| `export_llama3_to_torch_xla.py` | Convert a KerasHub Llama3 preset to a raw PyTorch checkpoint for XLA/TPU |
| `run_llama3_xla.py` | Run text generation from an XLA checkpoint on an XLA device |

---

## Prerequisites

```bash
pip install keras keras-hub transformers safetensors torch absl-py
```

For XLA/TPU scripts, also install `torch_xla` (version must match your `torch`):

```bash
pip install torch_xla[tpu]==<torch_version> \
    -f https://storage.googleapis.com/libtpu-releases/index.html
```

Set the Keras backend to PyTorch (required for all export scripts):

```bash
export KERAS_BACKEND=torch
```

---

## export_llama3_to_hf.py

Converts a KerasHub Llama3 model to HuggingFace Transformers format, writing:
- `config.json` — HF model configuration
- `model.safetensors` (or sharded `model-00001-of-NNNNN.safetensors`) — weights
- `tokenizer.json` — fast tokenizer (BPE, compatible with `PreTrainedTokenizerFast`)
- `tokenizer_config.json` — tokenizer metadata
- `generation_config.json` — generation defaults

### Supported Presets

| Preset | Description |
|--------|-------------|
| `llama3_8b_en` | Llama 3 8B base model |
| `llama3_8b_en_int8` | Llama 3 8B (int8 quantized) |
| `llama3_instruct_8b_en` | Instruction-tuned Llama 3 8B |
| `llama3_instruct_8b_en_int8` | Instruction-tuned Llama 3 8B (int8 quantized) |
| `llama3.1_8b` | Llama 3.1 8B base model |
| `llama3.1_instruct_8b` | Instruction-tuned Llama 3.1 8B |
| `llama3.1_guard_8b` | Llama Guard 3.1 8B |
| `llama3.2_1b` | Llama 3.2 1B base model |
| `llama3.2_instruct_1b` | Instruction-tuned Llama 3.2 1B |
| `llama3.2_3b` | Llama 3.2 3B base model |
| `llama3.2_instruct_3b` | Instruction-tuned Llama 3.2 3B |
| `llama3.2_guard_1b` | Llama Guard 3.2 1B |

### Usage

**Basic export (KerasHub preset → HF format):**

```bash
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_8b_en \
    --output_dir ./llama3_8b_hf/
```

**Export instruction-tuned variant:**

```bash
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_instruct_8b_en \
    --output_dir ./llama3_instruct_8b_hf/
```

**Export a fine-tuned checkpoint:**

```bash
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_8b_en \
    --weights_file my_finetuned_llama3.weights.h5 \
    --output_dir ./my_finetuned_llama3_hf/
```

**Export in bfloat16 (reduced memory usage):**

```bash
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_8b_en \
    --output_dir ./llama3_8b_hf/ \
    --dtype bfloat16
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | *required* | KerasHub preset name |
| `--weights_file` | `None` | Path to `.weights.h5` custom weights file |
| `--output_dir` | `llama3_hf` | Output directory for exported files |
| `--dtype` | `float32` | Model precision (`float32`, `float16`, `bfloat16`) |

### Loading the Exported Model

After export, load the model with HuggingFace Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./llama3_8b_hf/")
model = AutoModelForCausalLM.from_pretrained("./llama3_8b_hf/")

inputs = tokenizer("The quick brown fox", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **Note:** Llama3 uses a BPE tokenizer stored as `tokenizer.json`
> (a `PreTrainedTokenizerFast`). Unlike Mistral/Gemma, you do **not** need
> `use_fast=False` — simply use `AutoTokenizer.from_pretrained()`.

---

## verify_llama3_export.py

Loads both a KerasHub Llama3 model and a HuggingFace exported model and
checks that they produce matching outputs. Two types of verification:

1. **Logit comparison** — runs a forward pass through both models on the
   same token sequence and compares the logits at the last position.
2. **Generation comparison** — runs greedy decoding with both models and
   compares the generated text.

### Usage

**Quick verification (single prompt):**

```bash
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_8b_en \
    --hf_model_path ./llama3_8b_hf/
```

**Custom prompt:**

```bash
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_8b_en \
    --hf_model_path ./llama3_8b_hf/ \
    --test_prompt "Explain the theory of relativity in simple terms."
```

**Full test suite (5 prompts):**

```bash
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_8b_en \
    --hf_model_path ./llama3_8b_hf/ \
    --run_all_tests
```

**Relax tolerance for bfloat16 models:**

```bash
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_8b_en \
    --hf_model_path ./llama3_8b_hf/ \
    --logit_tolerance 1e-2
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--keras_preset` | *required* | KerasHub Llama3 preset name |
| `--hf_model_path` | *required* | Path to HuggingFace model directory |
| `--test_prompt` | `"The quick brown fox..."` | Prompt for single-prompt mode |
| `--max_length` | `50` | Max generation length in tokens |
| `--run_all_tests` | `False` | Run 5-prompt comprehensive suite |
| `--compare_logits` | `True` | Compare logits numerically |
| `--logit_tolerance` | `1e-3` | Max allowed absolute logit difference |

---

## Typical Workflow

### HuggingFace export

```bash
# 1. Export the model
python tools/llama3/export_llama3_to_hf.py \
    --preset llama3_8b_en \
    --output_dir ./llama3_8b_hf/

# 2. Verify the export
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_8b_en \
    --hf_model_path ./llama3_8b_hf/ \
    --run_all_tests

# 3. (Optional) Upload to HuggingFace Hub
#    huggingface-cli upload <your-org>/llama3-8b-keras-export ./llama3_8b_hf/
```

### XLA/TPU export

```bash
# 1. Convert to XLA checkpoint
python tools/llama3/export_llama3_to_torch_xla.py \
    --preset llama3_8b_en \
    --output_dir llama3_xla

# 2. Run inference on XLA device
python tools/llama3/run_llama3_xla.py \
    --checkpoint_dir llama3_xla \
    --prompt "The capital of France is"
```

---

## Troubleshooting

**`ValueError: Couldn't instantiate the backend tokenizer`**
This means the `tokenizer.json` is missing or malformed. Re-run the export
script to regenerate it. The Llama3 tokenizer uses BPE format, so the export
writes a full `tokenizer.json` compatible with the `tokenizers` library.

**Logits differ by more than tolerance**
- Check that both models use the same precision. Export with `--dtype float32`
  for highest fidelity.
- For bfloat16 models, use `--logit_tolerance 1e-2`.
- Ensure the `KERAS_BACKEND=torch` environment variable is set before running.

**`ModuleNotFoundError: No module named 'keras_hub'`**
Make sure you have installed KerasHub from the local source:
```bash
pip install -e /path/to/keras-hub
```

**`ImportError: torch_xla is required`**
Install `torch_xla` with a version matching your `torch` installation:
```bash
pip install torch_xla[tpu]==<your_torch_version> \
    -f https://storage.googleapis.com/libtpu-releases/index.html
```

**Out of memory**
Try exporting with `--dtype bfloat16` or use a machine with more RAM/VRAM.
For the 70B models, you will need multiple GPUs or a high-memory CPU setup.
