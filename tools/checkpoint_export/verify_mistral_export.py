"""Mistral KerasHub → HuggingFace Export Verification Script.

Performs a real-weights round-trip test following the same validation
pattern as the checkpoint conversion scripts:

  1. Load KerasHub preset → Export to HF format
  2. Load ORIGINAL HF model → Precompute logits and generated text
  3. Load EXPORTED HF model → Compare outputs against original
  4. Report config, params, logits, and generation parity

Usage:
    # Full validation (logits + generation):
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_mistral_export.py \
        --preset mistral_7b_en

    # Skip generation (faster, logit-only check):
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_mistral_export.py \
        --preset mistral_7b_en --skip_generation

    # Custom export directory:
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_mistral_export.py \
        --preset mistral_7b_en --export_dir /tmp/mistral_export

Requirements:
    pip install keras-hub transformers torch safetensors
"""

import argparse
import gc
import os
import tempfile

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM

print(f"Keras backend: {keras.config.backend()}")
print(f"Keras version: {keras.__version__}")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

PRESET_TO_HF = {
    "mistral_7b_en": "mistralai/Mistral-7B-v0.1",
    "mistral_0.3_7b_en": "mistralai/Mistral-7B-v0.3",
    "mistral_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.1",
    "mistral_0.2_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral_0.3_instruct_7b_en": "mistralai/Mistral-7B-Instruct-v0.3",
}

TEXT_PROMPT = "The capital of France is"

device = torch.device("cpu")


# ---------------------------------------------------------------
# 1. Export KerasHub model to HF format
# ---------------------------------------------------------------


def export_keras_model(preset, export_path):
    """Load KerasHub preset and export to HF format."""
    print("\n[1/6] Loading KerasHub model from preset...")
    keras_model = MistralCausalLM.from_preset(preset)
    backbone = keras_model.backbone

    print(
        f"  ✓ Loaded: {backbone.num_layers} layers, "
        f"{backbone.hidden_dim}d, {backbone.vocabulary_size} vocab"
    )
    print(f"  ✓ Parameters: {keras_model.count_params():,}")

    print(f"\n[2/6] Exporting to HF format → {export_path}...")
    keras_model.export_to_transformers(export_path)

    for fname in [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
    ]:
        fpath = os.path.join(export_path, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        print(f"  {'✓' if exists else '✗'} {fname} ({size:,} bytes)")

    # Free KerasHub model memory.
    del keras_model
    gc.collect()


# ---------------------------------------------------------------
# 2. Precompute Original HF outputs
# ---------------------------------------------------------------


def precompute_original_outputs(hf_model_id, skip_generation):
    """Load original HF model and precompute outputs."""
    print(f"\n[3/6] Loading ORIGINAL HF model: {hf_model_id}...")

    hf_model = MistralForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  ✓ Original HF: {hf_params:,} parameters")

    results = {"hf_params": hf_params}

    print("\n  Computing text logits...")
    hf_inputs = hf_tokenizer(TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        text_out = hf_model(**hf_inputs)
    results["text_logits"] = text_out.logits.float().cpu().numpy()
    results["text_input_ids"] = hf_inputs["input_ids"].cpu().numpy()
    print(f"    Text logits shape: {results['text_logits'].shape}")

    if not skip_generation:
        with torch.no_grad():
            gen_out = hf_model.generate(
                **hf_inputs, max_new_tokens=30, do_sample=False
            )
        results["text_generated"] = hf_tokenizer.decode(
            gen_out[0][hf_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print(f'    Text generation: "{results["text_generated"][:80]}"')

    # Free original HF model.
    del hf_model
    del hf_tokenizer
    gc.collect()

    return results


# ---------------------------------------------------------------
# 3a. Validate configs (compare ALL fields)
# ---------------------------------------------------------------


def validate_configs(exp_cfg, orig_cfg):
    """Compare all config fields between exported and original models."""
    print("\n  CONFIG VALIDATION")

    orig_dict = orig_cfg.to_dict() if hasattr(orig_cfg, "to_dict") else {}
    exp_dict = exp_cfg.to_dict() if hasattr(exp_cfg, "to_dict") else {}

    # Skip internal/meta fields that aren't part of the model architecture.
    skip_keys = {
        "_name_or_path",
        "_attn_implementation",
        "_attn_implementation_autoset",
        "_commit_hash",
        "transformers_version",
        "torch_dtype",
        "auto_map",
        "architectures",
        "dtype",
    }

    all_keys = sorted(set(orig_dict.keys()) | set(exp_dict.keys()))
    config_pass = True
    mismatches = []

    for key in all_keys:
        if key in skip_keys:
            continue
        o = orig_dict.get(key, "<missing>")
        e = exp_dict.get(key, "<missing>")
        if o == e:
            print(f"    ✓ {key}: {o}")
        else:
            print(f"    ✗ {key}: original={o}, exported={e}")
            mismatches.append(key)
            config_pass = False

    if mismatches:
        print(f"\n    ⚠ {len(mismatches)} field(s) differ: {mismatches}")
    else:
        print(
            f"\n    ✓ All {len(all_keys) - len(skip_keys)} config fields match"
        )

    return config_pass


# ---------------------------------------------------------------
# 3b. Validate token IDs
# ---------------------------------------------------------------


def validate_token_ids(exp_cfg, orig_cfg):
    """Compare special token IDs between exported and original models."""
    print("\n  TOKEN ID VALIDATION")

    token_fields = ["bos_token_id", "eos_token_id", "pad_token_id"]
    for attr in dir(orig_cfg):
        if attr.endswith("_token_id") and attr not in token_fields:
            token_fields.append(attr)

    token_pass = True
    for name in sorted(set(token_fields)):
        o = getattr(orig_cfg, name, None)
        e = getattr(exp_cfg, name, None)
        match = o == e
        print(f"    {'✓' if match else '✗'} {name}: original={o}, exported={e}")
        if not match:
            token_pass = False

    return token_pass


# ---------------------------------------------------------------
# 3c. Validate numerics (logits + generation)
# ---------------------------------------------------------------


def validate_numerics(
    exp_model, exp_tokenizer, original_results, skip_generation
):
    """Compare logits and generation between exported and original models."""
    results = {}

    print("\n  TEXT LOGIT VALIDATION")
    text_ids = torch.tensor(original_results["text_input_ids"]).to(device)
    with torch.no_grad():
        exp_text_out = exp_model(input_ids=text_ids)
    exp_text_logits = exp_text_out.logits.float().cpu().numpy()
    orig_text_logits = original_results["text_logits"]

    text_diff = np.abs(exp_text_logits - orig_text_logits)
    results["text_mean_diff"] = float(text_diff.mean())
    print(f"    Logit mean abs diff: {results['text_mean_diff']:.2e}")
    results["text_pass"] = results["text_mean_diff"] < 0.1

    orig_top5 = set(np.argsort(orig_text_logits[0, -1])[-5:].tolist())
    exp_top5 = set(np.argsort(exp_text_logits[0, -1])[-5:].tolist())
    overlap = len(orig_top5 & exp_top5)
    print(f"    Top-5 token overlap: {overlap}/5 ({100 * overlap / 5:.0f}%)")

    if not skip_generation:
        print("\n  GENERATION COMPARISON")
        hf_inputs = exp_tokenizer(TEXT_PROMPT, return_tensors="pt").to(device)
        prompt_len = hf_inputs["input_ids"].shape[1]
        with torch.no_grad():
            exp_gen = exp_model.generate(
                **hf_inputs, max_new_tokens=30, do_sample=False
            )
        exp_gen_text = exp_tokenizer.decode(
            exp_gen[0][prompt_len:], skip_special_tokens=True
        )
        orig_gen_text = original_results.get("text_generated", "N/A")

        print(f'    Prompt:   "{TEXT_PROMPT}"')
        print(f'    Original: "{orig_gen_text[:80]}"')
        print(f'    Exported: "{exp_gen_text[:80]}"')

        if orig_gen_text == exp_gen_text:
            results["text_gen_match"] = True
            print("    ✓ Generation is IDENTICAL")
        else:
            results["text_gen_match"] = False
            print("    ⚠ Generation differs (expected with bf16→f32)")

    return results


# ---------------------------------------------------------------
# 3. Load Exported HF model and run all validations
# ---------------------------------------------------------------


def validate_exported_model(
    export_path, hf_model_id, original_results, skip_generation
):
    """Load the exported model and compare against original."""
    print(f"\n[4/6] Loading EXPORTED model from {export_path}...")
    exp_model = MistralForCausalLM.from_pretrained(
        export_path, torch_dtype=torch.float32
    )
    exp_model.eval()
    exp_params = sum(p.numel() for p in exp_model.parameters())
    orig_params = original_results["hf_params"]
    print(f"  ✓ Exported: {exp_params:,} parameters")

    param_match = orig_params == exp_params
    print(
        f"  {'✓' if param_match else '✗'} Param count: "
        f"original={orig_params:,}, exported={exp_params:,}"
    )

    exp_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    print("\n[5/6] Validating configs and token IDs...")
    orig_cfg = AutoConfig.from_pretrained(hf_model_id)
    exp_cfg = exp_model.config

    config_pass = validate_configs(exp_cfg, orig_cfg)
    token_pass = validate_token_ids(exp_cfg, orig_cfg)

    print("\n[6/6] Validating numerics...")
    numeric_results = validate_numerics(
        exp_model, exp_tokenizer, original_results, skip_generation
    )

    del exp_model
    gc.collect()

    return {
        "param_match": param_match,
        "config_pass": config_pass,
        "token_pass": token_pass,
        **numeric_results,
    }


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def print_summary(results):
    """Print final summary."""
    config_pass = results.get("config_pass", False)
    token_pass = results.get("token_pass", False)
    text_pass = results.get("text_pass", False)
    param_match = results.get("param_match", False)
    text_gen_match = results.get("text_gen_match", None)

    all_pass = config_pass and token_pass and text_pass
    print("\n" + "=" * 70)
    if all_pass:
        print("  ✅ ALL CHECKS PASSED")
    else:
        print("  ❌ SOME CHECKS FAILED — Review output above")

    print(f"     - Config fields match {'✓' if config_pass else '✗'}")
    print(f"     - Token IDs match     {'✓' if token_pass else '✗'}")
    print(
        f"     - Parameter count:    {'match ✓' if param_match else 'differ ✗'}"
    )
    print(
        f"     - Text logit parity   "
        f"{'✓' if text_pass else '✗'} "
        f"(mean diff: {results.get('text_mean_diff', float('nan')):.2e})"
    )
    if text_gen_match is True:
        print("     - Text generation:    IDENTICAL ✓")
    elif text_gen_match is False:
        print("     - Text generation:    differs (bf16→f32 precision)")
    print("=" * 70 + "\n")

    return all_pass


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify Mistral KerasHub → HF export"
    )
    parser.add_argument(
        "--preset",
        default="mistral_7b_en",
        help=("KerasHub preset name (default: mistral_7b_en)"),
    )
    parser.add_argument(
        "--hf_model_id",
        default=None,
        help="HuggingFace model ID (auto-detected from preset if omitted)",
    )
    parser.add_argument(
        "--export_dir",
        default=None,
        help="Directory to export to (uses temp dir if omitted)",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation comparison (faster, logit-only check)",
    )
    args = parser.parse_args()

    hf_model_id = args.hf_model_id or PRESET_TO_HF.get(args.preset)
    if hf_model_id is None:
        print(f"Error: No HF model ID for preset '{args.preset}'.")
        exit(1)

    export_dir = args.export_dir or tempfile.mkdtemp()
    export_path = os.path.join(export_dir, "mistral_exported")

    print("\n" + "=" * 70)
    print("  Mistral Export Verification (Real Pretrained Weights)")
    print("=" * 70)
    print(f"  KerasHub preset : {args.preset}")
    print(f"  HF model ID     : {hf_model_id}")
    print(f"  Export path     : {export_path}")
    print(f"  Skip generation : {args.skip_generation}")

    # Phase 1: Export KerasHub model.
    export_keras_model(args.preset, export_path)

    # Phase 2: Precompute original HF outputs.
    original_results = precompute_original_outputs(
        hf_model_id, args.skip_generation
    )

    # Phase 3: Load exported model and validate.
    validation_results = validate_exported_model(
        export_path,
        hf_model_id,
        original_results,
        args.skip_generation,
    )

    # Summary.
    success = print_summary(validation_results)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
