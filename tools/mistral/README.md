# Mistral Export and Verification Tools

This directory contains tools for exporting KerasHub Mistral models to HuggingFace Transformers format and verifying the conversion.

## Scripts

### 1. `export_mistral_to_hf.py`

Converts KerasHub Mistral models to HuggingFace Transformers format with safetensors weights.

**Features:**
- Export from KerasHub presets
- Export custom fine-tuned models
- Support for all Mistral variants (7B, instruct versions, etc.)
- Configurable precision (float32, float16, bfloat16)
- Automatic tokenizer conversion

**Usage:**

Export a preset model:
```bash
python tools/mistral/export_mistral_to_hf.py \
    --preset mistral_7b_en \
    --output_dir ./mistral_hf/
```

Export with custom weights:
```bash
python tools/mistral/export_mistral_to_hf.py \
    --preset mistral_7b_en \
    --weights_file ./fine_tuned_mistral.weights.h5 \
    --output_dir ./mistral_hf/
```

Export with different precision:
```bash
python tools/mistral/export_mistral_to_hf.py \
    --preset mistral_instruct_7b_en \
    --output_dir ./mistral_hf/ \
    --dtype float16
```

**Available Presets:**
- `mistral_7b_en` - Base Mistral 7B model
- `mistral_0.3_7b_en` - Mistral 7B v0.3
- `mistral_instruct_7b_en` - Instruction-tuned Mistral 7B
- `mistral_0.2_instruct_7b_en` - Instruction-tuned v0.2
- `mistral_0.3_instruct_7b_en` - Instruction-tuned v0.3

### 2. `verify_mistral_export.py`

Verifies that exported models generate correct outputs by comparing KerasHub and HuggingFace implementations.

**Features:**
- Generation comparison (text output)
- Logit-level numerical verification
- Multiple test prompt support
- Configurable tolerance levels
- Comprehensive test suite

**Usage:**

Basic verification:
```bash
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_7b_en \
    --hf_model_path ./mistral_hf/ \
    --test_prompt "What is your favorite"
```

Run comprehensive tests:
```bash
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_7b_en \
    --hf_model_path ./mistral_hf/ \
    --run_all_tests
```

Custom test with longer generation:
```bash
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_instruct_7b_en \
    --hf_model_path ./mistral_hf/ \
    --test_prompt "Explain the theory of relativity" \
    --max_length 100
```

Skip logit comparison (faster):
```bash
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_7b_en \
    --hf_model_path ./mistral_hf/ \
    --compare_logits=false
```

## Complete Workflow Example

Here's a complete example of exporting and verifying a Mistral model:

```bash
# 1. Export the model
python tools/mistral/export_mistral_to_hf.py \
    --preset mistral_7b_en \
    --output_dir ./mistral_7b_hf/ \
    --dtype float32

# 2. Verify the export
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_7b_en \
    --hf_model_path ./mistral_7b_hf/ \
    --run_all_tests

# 3. Use with HuggingFace Transformers
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./mistral_7b_hf/')
tokenizer = AutoTokenizer.from_pretrained('./mistral_7b_hf/', use_fast=False)

inputs = tokenizer('Hello, my name is', return_tensors='pt')
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

## Requirements

These scripts require the following packages:

```bash
pip install keras-hub
pip install transformers
pip install torch
pip install safetensors
pip install absl-py
```

## Output Format

The exported model will be in standard HuggingFace format with the following files:

```
mistral_hf/
├── config.json              # Model configuration
├── model.safetensors        # Model weights
├── tokenizer.model          # SentencePiece tokenizer
└── tokenizer_config.json    # Tokenizer configuration
```

## Verification Details

The verification script performs the following checks:

1. **Generation Test**: Compares text outputs from both models with the same prompt
2. **Logit Verification**: Numerically compares output logits to ensure weight conversion is correct
3. **Top-K Prediction**: Verifies that the most likely next tokens match
4. **Shape Validation**: Ensures output dimensions are consistent

**Expected Results:**
- ✅ Exact or very similar text outputs (minor differences due to tokenization are acceptable)
- ✅ Logits within tolerance (default: 1e-3)
- ✅ Top-5 token predictions match

## Troubleshooting

### Export Issues

**"Cannot load preset"**
- Ensure the preset name is correct (see Available Presets above)
- Check that you have downloaded the preset: `keras_hub.models.MistralCausalLM.from_preset(preset)`

**"Invalid weights file"**
- Weights file must have `.weights.h5` extension
- Ensure the file is a valid Keras weights file

### Verification Issues

**"Outputs differ significantly"**
- This may be due to tokenization differences (acceptable)
- Check if one output contains the other
- Run logit comparison for numerical verification

**"Logits exceed tolerance"**
- May indicate an issue with weight conversion
- Try increasing tolerance with `--logit_tolerance`
- Check dtype consistency between models

**"Top-K predictions differ"**
- Minor differences in logits can cause different token rankings
- If text outputs are similar, this is usually acceptable

## Notes

- Export uses the PyTorch backend (`KERAS_BACKEND=torch`)
- Models are exported with safetensors for security and efficiency
- The verification script uses greedy decoding (no sampling) for reproducibility
- SentencePiece tokenizer requires `use_fast=False` when loading with HuggingFace

## Contributing

When adding new Mistral variants or features:

1. Update `PRESET_MAP` in both scripts
2. Test export and verification with the new variant
3. Update this README with new presets
4. Add any variant-specific configuration handling

## Related Files

- Core export implementation: `keras_hub/src/utils/transformers/export/mistral.py`
- Export tests: `keras_hub/src/utils/transformers/export/mistral_test.py`
- Import implementation: `keras_hub/src/utils/transformers/convert_mistral.py`
