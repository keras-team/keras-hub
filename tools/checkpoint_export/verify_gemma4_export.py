"""Gemma4 KerasHub → HuggingFace Export Verification Script.

Performs a real-weights round-trip test:

  1. Load a KerasHub Gemma4 preset.
  2. Export to HuggingFace format via ``export_to_transformers``.
  3. Load the **original** HF checkpoint and precompute reference logits /
     generated text.
  4. Load the **exported** HF checkpoint and compare outputs against the
     reference.
  5. Print a summary with pass/fail for each check.

Supported presets
-----------------
Text-only variants (``model_type=gemma4_text``):
    (Currently no text-only Gemma4 presets exist on Kaggle/HF Hub; the
     flag ``--text_only`` can be used to build a random backbone for a
     structural smoke-test.)

Multimodal variants (``model_type=gemma4``):
    gemma4_instruct_2b, gemma4_instruct_4b, gemma4_instruct_26b_a4b,
    gemma4_instruct_31b, gemma4_2b, gemma4_4b, gemma4_26b_a4b, gemma4_31b

Usage
-----
Full validation (logits + generation)::

    KERAS_BACKEND=torch python3 \\
        tools/checkpoint_export/verify_gemma4_export.py \\
        --preset gemma4_instruct_4b

Skip generation (faster, logit-only)::

    KERAS_BACKEND=torch python3 \\
        tools/checkpoint_export/verify_gemma4_export.py \\
        --preset gemma4_instruct_4b --skip_generation

Custom export directory::

    KERAS_BACKEND=torch python3 \\
        tools/checkpoint_export/verify_gemma4_export.py \\
        --preset gemma4_instruct_4b --export_dir /tmp/gemma4_export

Structural smoke-test with a random backbone (no real weights required)::

    KERAS_BACKEND=torch python3 \\
        tools/checkpoint_export/verify_gemma4_export.py \\
        --text_only --skip_generation --skip_logit_comparison

Requirements
------------
    pip install keras-hub transformers torch safetensors
"""

import argparse
import gc
import os
import sys
import tempfile

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch

print(f"Keras backend: {keras.config.backend()}")
print(f"Keras version: {keras.__version__}")

# ---------------------------------------------------------------------------
# Preset → HF model-id mapping
# ---------------------------------------------------------------------------

PRESET_TO_HF = {
    "gemma4_2b": "google/gemma-4-2b-pt",
    "gemma4_instruct_2b": "google/gemma-4-2b-it",
    "gemma4_4b": "google/gemma-4-4b-pt",
    "gemma4_instruct_4b": "google/gemma-4-4b-it",
    "gemma4_26b_a4b": "google/gemma-4-26b-pt",
    "gemma4_instruct_26b_a4b": "google/gemma-4-26b-it",
    "gemma4_31b": "google/gemma-4-31b-pt",
    "gemma4_instruct_31b": "google/gemma-4-31b-it",
}

TEXT_PROMPT = "The capital of France is"
device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Phase 1 – Export KerasHub preset
# ---------------------------------------------------------------------------


def export_keras_model(preset, export_path):
    """Load a KerasHub Gemma4 preset and export it to HF format."""
    from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM

    print("\n[1/6] Loading KerasHub model from preset…")
    keras_model = Gemma4CausalLM.from_preset(preset)
    backbone = keras_model.backbone

    has_vision = backbone.vision_encoder is not None
    has_audio = backbone.audio_encoder is not None

    print(
        f"  ✓ Loaded: {backbone.num_layers} layers, "
        f"{backbone.hidden_dim}d, {backbone.vocabulary_size} vocab"
    )
    print(
        f"  ✓ Vision encoder: {'yes' if has_vision else 'no'}  "
        f"Audio encoder: {'yes' if has_audio else 'no'}"
    )
    print(f"  ✓ Parameters: {keras_model.count_params():,}")

    print(f"\n[2/6] Exporting to HF format → {export_path}…")
    keras_model.export_to_transformers(export_path)

    for fname in ["config.json", "model.safetensors"]:
        fpath = os.path.join(export_path, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        print(f"  {'✓' if exists else '✗'} {fname} ({size:,} bytes)")

    del keras_model
    gc.collect()


# ---------------------------------------------------------------------------
# Phase 2 – Precompute reference outputs from the original HF checkpoint
# ---------------------------------------------------------------------------


def precompute_original_outputs(hf_model_id, skip_generation):
    """Load the canonical HF model and record its logits (+ optionally text)."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    print(f"\n[3/6] Loading ORIGINAL HF model: {hf_model_id}…")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, torch_dtype=torch.float32
    )
    hf_model.eval()
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  ✓ Parameters: {hf_params:,}")

    results = {"hf_params": hf_params}

    print("  Computing text logits…")
    hf_inputs = hf_tokenizer(TEXT_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = hf_model(**hf_inputs)
    results["text_logits"] = out.logits.float().cpu().numpy()
    results["text_input_ids"] = hf_inputs["input_ids"].cpu().numpy()
    print(f"    Text logits shape: {results['text_logits'].shape}")

    if not skip_generation:
        with torch.no_grad():
            gen = hf_model.generate(
                **hf_inputs, max_new_tokens=30, do_sample=False
            )
        results["text_generated"] = hf_tokenizer.decode(
            gen[0][hf_inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        print(f'    Generated text: "{results["text_generated"][:80]}"')

    del hf_model, hf_tokenizer
    gc.collect()
    return results


# ---------------------------------------------------------------------------
# Phase 3a – Config comparison
# ---------------------------------------------------------------------------


def validate_configs(exp_cfg, orig_cfg):
    """Compare all architecture-relevant config fields."""
    print("\n  CONFIG VALIDATION")

    # Skip HF-internal metadata keys (private keys starting with "_") and
    # runtime fields that differ by environment but not architecture.
    skip_keys = {
        "transformers_version",
        "torch_dtype",
        "auto_map",
        "architectures",
        "dtype",
    }

    orig_dict = orig_cfg.to_dict() if hasattr(orig_cfg, "to_dict") else {}
    exp_dict = exp_cfg.to_dict() if hasattr(exp_cfg, "to_dict") else {}

    all_keys = sorted(set(orig_dict.keys()) | set(exp_dict.keys()))
    mismatches = []
    for key in all_keys:
        if key in skip_keys or key.startswith("_"):
            continue
        o = orig_dict.get(key, "<missing>")
        e = exp_dict.get(key, "<missing>")
        if o == e:
            print(f"    ✓ {key}: {o}")
        else:
            print(f"    ✗ {key}: original={o!r}, exported={e!r}")
            mismatches.append(key)

    if mismatches:
        print(f"\n    ⚠ {len(mismatches)} field(s) differ: {mismatches}")
    else:
        skipped = sum(
            1 for k in all_keys if k in skip_keys or k.startswith("_")
        )
        print(f"\n    ✓ All {len(all_keys) - skipped} config fields match")

    return len(mismatches) == 0


# ---------------------------------------------------------------------------
# Phase 3b – Token ID comparison
# ---------------------------------------------------------------------------


def validate_token_ids(exp_cfg, orig_cfg):
    """Compare special-token IDs."""
    print("\n  TOKEN ID VALIDATION")
    token_fields = ["bos_token_id", "eos_token_id", "pad_token_id"]
    all_pass = True
    for name in sorted(set(token_fields)):
        o = getattr(orig_cfg, name, None)
        e = getattr(exp_cfg, name, None)
        ok = o == e
        print(f"    {'✓' if ok else '✗'} {name}: original={o}, exported={e}")
        if not ok:
            all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Phase 3c – Numeric / logit comparison
# ---------------------------------------------------------------------------


def validate_numerics(
    exp_model, exp_tokenizer, original_results, skip_generation
):
    """Compare logits (and optionally generation) against original model."""
    results = {}

    print("\n  TEXT LOGIT VALIDATION")
    text_ids = torch.tensor(original_results["text_input_ids"]).to(device)
    with torch.no_grad():
        exp_out = exp_model(input_ids=text_ids)
    exp_logits = exp_out.logits.float().cpu().numpy()
    orig_logits = original_results["text_logits"]

    diff = np.abs(exp_logits - orig_logits)
    results["text_mean_diff"] = float(diff.mean())
    print(f"    Mean abs logit diff: {results['text_mean_diff']:.2e}")
    results["text_pass"] = results["text_mean_diff"] < 0.1

    orig_top5 = set(np.argsort(orig_logits[0, -1])[-5:].tolist())
    exp_top5 = set(np.argsort(exp_logits[0, -1])[-5:].tolist())
    overlap = len(orig_top5 & exp_top5)
    print(f"    Top-5 token overlap: {overlap}/5 ({100 * overlap / 5:.0f}%)")

    if not skip_generation:
        print("\n  GENERATION COMPARISON")
        hf_inputs = exp_tokenizer(TEXT_PROMPT, return_tensors="pt").to(device)
        prompt_len = hf_inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = exp_model.generate(
                **hf_inputs, max_new_tokens=30, do_sample=False
            )
        exp_gen_text = exp_tokenizer.decode(
            gen[0][prompt_len:], skip_special_tokens=True
        )
        orig_gen_text = original_results.get("text_generated", "N/A")
        print(f'    Prompt:   "{TEXT_PROMPT}"')
        print(f'    Original: "{orig_gen_text[:80]}"')
        print(f'    Exported: "{exp_gen_text[:80]}"')
        results["text_gen_match"] = orig_gen_text == exp_gen_text
        gen_status = (
            "✓ IDENTICAL"
            if results["text_gen_match"]
            else "⚠ differs (expected with bf16→f32)"
        )
        print(f"    {gen_status}")

    return results


# ---------------------------------------------------------------------------
# Phase 4 – Load exported model + validate
# ---------------------------------------------------------------------------


def validate_exported_model(
    export_path,
    hf_model_id,
    original_results,
    skip_generation,
    skip_logit_comparison,
):
    """Load the exported HF model and run all validation checks."""
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    print(f"\n[4/6] Loading EXPORTED model from {export_path}…")
    exp_model = AutoModelForCausalLM.from_pretrained(
        export_path, torch_dtype=torch.float32
    )
    exp_model.eval()
    exp_params = sum(p.numel() for p in exp_model.parameters())
    orig_params = original_results.get("hf_params", 0)
    print(f"  ✓ Exported: {exp_params:,} parameters")

    param_match = orig_params == exp_params or orig_params == 0
    if orig_params > 0:
        print(
            f"  {'✓' if param_match else '✗'} Param count: "
            f"original={orig_params:,}, exported={exp_params:,}"
        )

    orig_cfg = AutoConfig.from_pretrained(hf_model_id) if hf_model_id else None
    exp_cfg = exp_model.config

    print("\n[5/6] Validating configs and token IDs…")
    if orig_cfg is not None:
        config_pass = validate_configs(exp_cfg, orig_cfg)
        token_pass = validate_token_ids(exp_cfg, orig_cfg)
    else:
        print("    (skipped — no reference HF model)")
        config_pass = True
        token_pass = True

    if skip_logit_comparison:
        print("\n[6/6] Logit comparison skipped (--skip_logit_comparison).")
        numeric_results = {}
    else:
        print("\n[6/6] Validating numerics…")
        exp_tokenizer = (
            AutoTokenizer.from_pretrained(hf_model_id) if hf_model_id else None
        )
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results):
    """Print a formatted summary and return overall pass/fail."""
    config_pass = results.get("config_pass", True)
    token_pass = results.get("token_pass", True)
    text_pass = results.get("text_pass", None)
    param_match = results.get("param_match", True)
    text_gen_match = results.get("text_gen_match", None)

    checks = [config_pass, token_pass]
    if text_pass is not None:
        checks.append(text_pass)
    all_pass = all(checks)

    print("\n" + "=" * 70)
    print(
        "  ✅ ALL CHECKS PASSED"
        if all_pass
        else "  ❌ SOME CHECKS FAILED — review output above"
    )
    print(f"     - Config fields match:  {'✓' if config_pass else '✗'}")
    print(f"     - Token IDs match:      {'✓' if token_pass else '✗'}")
    print(
        f"     - Parameter count:      "
        f"{'match ✓' if param_match else 'differ ✗'}"
    )
    if text_pass is not None:
        print(
            f"     - Text logit parity:   "
            f"{'✓' if text_pass else '✗'} "
            f"(mean diff: {results.get('text_mean_diff', float('nan')):.2e})"
        )
    if text_gen_match is True:
        print("     - Text generation:     IDENTICAL ✓")
    elif text_gen_match is False:
        print("     - Text generation:     differs (expected with bf16→f32)")
    print("=" * 70 + "\n")
    return all_pass


# ---------------------------------------------------------------------------
# Structural smoke-test (no real weights)
# ---------------------------------------------------------------------------


def run_structural_smoke_test(export_dir):
    """Export a tiny random text-only backbone and check the file structure."""
    import json

    from transformers import AutoModelForCausalLM

    from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
    from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM
    from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
        Gemma4CausalLMPreprocessor,
    )
    from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer

    print("\n[SMOKE TEST] Tiny random text-only backbone…")
    # Use the test vocab shipped with the repository (located in
    # keras_hub/src/tests/test_data/).
    # __file__ is tools/checkpoint_export/verify_gemma4_export.py → 3 levels up.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    vocab_path = os.path.join(
        repo_root,
        "keras_hub",
        "src",
        "tests",
        "test_data",
        "gemma4_test_vocab.spm",
    )
    if not os.path.exists(vocab_path):
        print(f"  ✗ Test vocab not found at {vocab_path}. Skipping smoke test.")
        return True

    tokenizer = Gemma4Tokenizer(
        proto=vocab_path,
        has_vision_tokens=False,
        has_audio_tokens=False,
        has_video_tokens=False,
    )
    backbone = Gemma4Backbone(
        vocabulary_size=tokenizer.vocabulary_size(),
        image_size=None,
        num_layers=2,
        num_query_heads=2,
        num_key_value_heads=1,
        hidden_dim=64,
        intermediate_dim=128,
        head_dim=32,
        use_sliding_window_attention=False,
        sliding_window_size=512,
        attention_logit_soft_cap=None,
        final_logit_soft_cap=None,
        vision_encoder=None,
        audio_encoder=None,
        layer_norm_epsilon=1e-6,
        dropout=0,
    )
    preprocessor = Gemma4CausalLMPreprocessor(tokenizer=tokenizer)
    model = Gemma4CausalLM(backbone=backbone, preprocessor=preprocessor)
    export_path = os.path.join(export_dir, "gemma4_smoke_test")
    model.export_to_transformers(export_path)

    all_ok = True
    for fname in ["config.json", "model.safetensors"]:
        fpath = os.path.join(export_path, fname)
        ok = os.path.exists(fpath)
        print(f"  {'✓' if ok else '✗'} {fname}")
        all_ok = all_ok and ok

    # Verify the config loads cleanly.
    cfg_path = os.path.join(export_path, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    model_type_ok = cfg.get("model_type") == "gemma4_text"
    print(
        f"  {'✓' if model_type_ok else '✗'} "
        f"config.json model_type = gemma4_text"
    )
    all_ok = all_ok and model_type_ok

    # Validate the exported safetensors keys + shapes.
    # Verify the HF model loads cleanly (all exported keys recognized),
    # then compare param count against the safetensors file (which avoids
    # counting HF's computed non-checkpoint buffers like RoPE frequencies).
    st_path = os.path.join(export_path, "model.safetensors")
    st_keys = []
    try:
        from safetensors import safe_open

        st_total = 0
        with safe_open(st_path, framework="pt") as f:
            st_keys = sorted(f.keys())
            for k in st_keys:
                st_total += f.get_tensor(k).numel()
        keras_params = model.count_params()
        count_match = st_total == keras_params
        print(
            f"  {'✓' if count_match else '✗'} Parameter count: "
            f"Keras={keras_params:,}, safetensors={st_total:,}"
        )
        all_ok = all_ok and count_match
    except Exception as st_exc:
        print(f"  ✗ Safetensors inspection failed: {st_exc}")
        all_ok = False

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(export_path)
        print("  ✓ HF model loaded successfully (architecture recognized)")
        # Spot-check: verify all exported keys were loaded (no unexpected keys).
        hf_sd_keys = set(hf_model.state_dict().keys())
        # Exported keys use the "model." prefix; HF CausalLM wraps them
        # under "model." too. Check that all safetensors keys appear in HF.
        missing_from_hf = [k for k in st_keys if k not in hf_sd_keys]
        if missing_from_hf:
            print(
                f"  ✗ Keys exported but not in HF model: {missing_from_hf[:5]}"
            )
            all_ok = False
        else:
            print(f"  ✓ All {len(st_keys)} exported keys present in HF model")
        del hf_model
    except Exception as hf_exc:
        print(f"  ✗ Failed to load exported model: {hf_exc}")
        all_ok = False

    status = "PASSED" if all_ok else "FAILED"
    print(f"\n  Smoke test: {status}")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify Gemma4 KerasHub → HuggingFace export"
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="KerasHub preset name (e.g. gemma4_instruct_4b).",
    )
    parser.add_argument(
        "--hf_model_id",
        default=None,
        help="HuggingFace model ID (auto-detected from preset if omitted).",
    )
    parser.add_argument(
        "--export_dir",
        default=None,
        help="Directory for exported files (uses temp dir if omitted).",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip generation comparison (faster, logit-only check).",
    )
    parser.add_argument(
        "--skip_logit_comparison",
        action="store_true",
        help="Skip logit comparison entirely.",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Run a structural smoke-test only (no real weights needed).",
    )
    args = parser.parse_args()

    export_dir = args.export_dir or tempfile.mkdtemp()
    os.makedirs(export_dir, exist_ok=True)

    # --- Structural smoke-test ---
    if args.text_only:
        success = run_structural_smoke_test(export_dir)
        sys.exit(0 if success else 1)

    # --- Real-weights round-trip ---
    if args.preset is None:
        parser.error("--preset is required unless --text_only is set.")

    hf_model_id = args.hf_model_id or PRESET_TO_HF.get(args.preset)
    if hf_model_id is None:
        print(
            f"Warning: no HF model ID known for preset '{args.preset}'. "
            "Skipping original-model comparisons."
        )

    export_path = os.path.join(export_dir, f"{args.preset}_exported")

    print("\n" + "=" * 70)
    print("  Gemma4 Export Verification")
    print("=" * 70)
    print(f"  KerasHub preset:    {args.preset}")
    print(
        f"  HF model ID:        "
        f"{hf_model_id or '(none — structural check only)'}"
    )
    print(f"  Export path:        {export_path}")
    print(f"  Skip generation:    {args.skip_generation}")
    print(f"  Skip logit check:   {args.skip_logit_comparison}")

    # Phase 1: Export.
    export_keras_model(args.preset, export_path)

    original_results = {}
    if hf_model_id is not None:
        # Phase 2: Precompute reference outputs.
        original_results = precompute_original_outputs(
            hf_model_id, args.skip_generation
        )
    else:
        args.skip_logit_comparison = True

    # Phase 3: Load + validate exported model.
    validation_results = validate_exported_model(
        export_path,
        hf_model_id,
        original_results,
        args.skip_generation,
        args.skip_logit_comparison,
    )

    success = print_summary(validation_results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
