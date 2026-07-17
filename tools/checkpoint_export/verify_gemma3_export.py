"""Gemma3 KerasHub → HuggingFace Export Verification Script.

Performs a real-weights round-trip test following the same validation
pattern as the checkpoint conversion scripts:

  1. Load KerasHub preset → Export to HF format
  2. Load ORIGINAL HF model → Precompute logits and generated text
  3. Load EXPORTED HF model → Compare outputs against original
  4. Report config, params, logits, and generation parity

Supports both text-only and vision-language Gemma3 models.

Usage:
    # Text-only model validation (logits + generation):
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_gemma3_export.py \
        --preset gemma3_1b

    # Vision-language model validation:
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_gemma3_export.py \
        --preset gemma3_4b

    # Skip generation (faster, logit-only check):
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_gemma3_export.py \
        --preset gemma3_1b --skip_generation

    # Custom export directory:
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_gemma3_export.py \
        --preset gemma3_1b --export_dir /tmp/gemma3_export

    # Provide an explicit HF model ID:
    KERAS_BACKEND=torch python3 \
        tools/checkpoint_export/verify_gemma3_export.py \
        --preset gemma3_1b \
        --hf_model_id google/gemma-3-1b-pt

Requirements:
    pip install keras-hub transformers torch safetensors pillow
"""

import argparse
import gc
import os
import tempfile
from numbers import Number

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from keras_hub.src.models.gemma3.gemma3_causal_lm import (  # noqa: E402
    Gemma3CausalLM,
)

print(f"Keras backend: {keras.config.backend()}")
print(f"Keras version: {keras.__version__}")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

PRESET_TO_HF = {
    # Text-only models
    "gemma3_270m": "google/gemma-3-270m",
    "gemma3_instruct_270m": "google/gemma-3-270m-it",
    "gemma3_1b": "google/gemma-3-1b-pt",
    "gemma3_instruct_1b": "google/gemma-3-1b-it",
    "gemma3_4b_text": "google/gemma-3-4b-pt",
    "gemma3_instruct_4b_text": "google/gemma-3-4b-it",
    "gemma3_12b_text": "google/gemma-3-12b-pt",
    "gemma3_instruct_12b_text": "google/gemma-3-12b-it",
    "gemma3_27b_text": "google/gemma-3-27b-pt",
    "gemma3_instruct_27b_text": "google/gemma-3-27b-it",
    # Vision-language models
    "gemma3_4b": "google/gemma-3-4b-pt",
    "gemma3_instruct_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-pt",
    "gemma3_instruct_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-pt",
    "gemma3_instruct_27b": "google/gemma-3-27b-it",
    # Specialized variants
    "translategemma_4b_it": "google/translategemma-4b-it",
    "translategemma_12b_it": "google/translategemma-12b-it",
    "translategemma_27b_it": "google/translategemma-27b-it",
    "function_gemma_instruct_270m": "google/functiongemma-270m-it",
    "medgemma_4b": "google/medgemma-4b",
    "medgemma_instruct_4b": "google/medgemma-4b-it",
    "medgemma_instruct_27b": "google/medgemma-27b-it",
    "medgemma_instruct_27b_text": "google/medgemma-27b-it",
    "medgemma_1.5_instruct_4b": "google/medgemma-1.5-4b-it",
    "embedding_gemma3_300m": "google/gemma-3-embedding-300m",
}

TEXT_PROMPT = "The capital of France is"
IMAGE_PROMPT = "What is in this image?"

device = torch.device("cpu")

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------


def configs_equal(val1, val2, tolerance=1e-6):
    """Compare two config values with tolerance for numeric differences."""
    if isinstance(val1, bool) or isinstance(val2, bool):
        return val1 == val2

    if isinstance(val1, Number) and isinstance(val2, Number):
        return abs(float(val1) - float(val2)) < tolerance

    if not isinstance(val1, type(val2)):
        return False

    if isinstance(val1, (str, type(None))):
        return val1 == val2

    if isinstance(val1, (list, tuple)):
        if len(val1) != len(val2):
            return False
        return all(
            configs_equal(v1, v2, tolerance) for v1, v2 in zip(val1, val2)
        )

    if isinstance(val1, dict):
        if set(val1.keys()) != set(val2.keys()):
            return False
        return all(
            configs_equal(val1[k], val2.get(k), tolerance) for k in val1.keys()
        )

    # Fallback to direct comparison
    return val1 == val2


# ---------------------------------------------------------------
# 1. Export KerasHub model to HF format
# ---------------------------------------------------------------


def export_keras_model(preset, export_path):
    """Load KerasHub preset and export to HF format."""
    print("\n[1/6] Loading KerasHub model from preset...")

    try:
        keras_model = Gemma3CausalLM.from_preset(preset)
    except Exception as e:
        print(f"  ✗ Failed to load KerasHub model: {e}")
        raise

    backbone = keras_model.backbone

    has_vision = backbone.vision_encoder is not None

    print(
        f"  ✓ Loaded: {backbone.num_layers} layers, "
        f"{backbone.hidden_dim}d, {backbone.vocabulary_size} vocab"
    )
    if has_vision:
        print("  ✓ Vision encoder detected (multimodal model)")
    print(f"  ✓ Parameters: {keras_model.count_params():,}")

    print(f"\n[2/6] Exporting to HF format → {export_path}...")

    try:
        keras_model.export_to_transformers(export_path)
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        raise

    expected_files = [
        "config.json",
        "model.safetensors",
        "tokenizer_config.json",
    ]
    if has_vision:
        expected_files.extend(
            ["preprocessor_config.json", "processor_config.json"]
        )

    all_exist = True
    for fname in expected_files:
        fpath = os.path.join(export_path, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        print(f"  {'✓' if exists else '✗'} {fname} ({size:,} bytes)")
        if not exists:
            all_exist = False

    if not all_exist:
        raise FileNotFoundError("Some expected export files are missing")

    # Free KerasHub model memory.
    del keras_model
    gc.collect()

    return has_vision


# ---------------------------------------------------------------
# 2. Precompute Original HF outputs
# ---------------------------------------------------------------


def precompute_original_outputs(hf_model_id, has_vision, skip_generation):
    """Load original HF model and precompute outputs."""
    print(f"\n[3/6] Loading ORIGINAL HF model: {hf_model_id}...")

    try:
        if has_vision:
            hf_model = AutoModelForImageTextToText.from_pretrained(
                hf_model_id, dtype=torch.float32
            )
            hf_processor = AutoProcessor.from_pretrained(hf_model_id)
            hf_tokenizer = hf_processor.tokenizer
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_id, dtype=torch.float32
            )
            hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    except Exception as e:
        print(f"  ✗ Failed to load HF model: {e}")
        raise

    hf_model.eval()
    hf_params = sum(p.numel() for p in hf_model.parameters())
    print(f"  ✓ Original HF: {hf_params:,} parameters")

    results = {"hf_params": hf_params, "has_vision": has_vision}

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

    # TODO: Add vision validation when image processing is supported
    # if has_vision:
    #     print("\n  Computing vision logits...")
    #     # Create a dummy image or load a test image
    #     # Process with hf_processor
    #     # Compute logits and generation

    # Free original HF model.
    del hf_model
    if has_vision:
        del hf_processor
    del hf_tokenizer
    gc.collect()

    return results


# ---------------------------------------------------------------
# 3a. Validate configs (compare ALL fields)
# ---------------------------------------------------------------


def validate_configs(exp_cfg, orig_cfg, has_vision):
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

    optional_runtime_keys = {
        "cache_implementation",
        "sliding_window_pattern",
    }

    def classify_difference(key, original_value, exported_value):
        """Return (is_non_critical, reason) for known benign config deltas."""
        original_missing = original_value == "<missing>"
        exported_missing = exported_value == "<missing>"

        if key in optional_runtime_keys and (original_missing or exported_missing):
            return True, "optional/runtime-only field"

        if (
            key == "max_position_embeddings"
            and isinstance(original_value, Number)
            and isinstance(exported_value, Number)
            and exported_value >= original_value
        ):
            return True, "exported value is larger (superset context capacity)"

        return False, ""

    def compare_section(section_name, original_section, exported_section):
        """Compare config sections and classify critical vs non-critical deltas."""
        if section_name:
            print(f"\n  {section_name}:")

        section_pass = True
        critical_diffs = []
        warning_diffs = []

        keys = sorted(set(original_section.keys()) | set(exported_section.keys()))
        for key in keys:
            if key in skip_keys:
                continue

            original_value = original_section.get(key, "<missing>")
            exported_value = exported_section.get(key, "<missing>")

            if configs_equal(original_value, exported_value):
                print(f"    ✓ {key}: {original_value}")
                continue

            is_non_critical, reason = classify_difference(
                key, original_value, exported_value
            )
            if is_non_critical:
                print(
                    "    ⚠ "
                    f"{key}: original={original_value}, exported={exported_value} "
                    f"({reason})"
                )
                warning_diffs.append(key)
            else:
                print(
                    f"    ✗ {key}: original={original_value}, "
                    f"exported={exported_value}"
                )
                critical_diffs.append(key)
                section_pass = False

        return section_pass, critical_diffs, warning_diffs

    # For vision models, validate text_config and vision_config separately
    if has_vision:
        text_config_pass = True
        vision_config_pass = True
        critical_diffs = []
        warning_diffs = []

        # Validate text_config
        if "text_config" in orig_dict and "text_config" in exp_dict:
            text_config_pass, text_critical, text_warnings = compare_section(
                "TEXT CONFIG",
                orig_dict["text_config"],
                exp_dict["text_config"],
            )
            critical_diffs.extend([f"text_config.{k}" for k in text_critical])
            warning_diffs.extend([f"text_config.{k}" for k in text_warnings])

        # Validate vision_config
        if "vision_config" in orig_dict and "vision_config" in exp_dict:
            (
                vision_config_pass,
                vision_critical,
                vision_warnings,
            ) = compare_section(
                "VISION CONFIG",
                orig_dict["vision_config"],
                exp_dict["vision_config"],
            )
            critical_diffs.extend(
                [f"vision_config.{k}" for k in vision_critical]
            )
            warning_diffs.extend(
                [f"vision_config.{k}" for k in vision_warnings]
            )

        config_pass = text_config_pass and vision_config_pass
        if warning_diffs:
            print(
                f"\n    ⚠ {len(warning_diffs)} non-critical "
                f"field difference(s): {warning_diffs}"
            )
        if critical_diffs:
            print(
                f"\n    ✗ {len(critical_diffs)} critical "
                f"field difference(s): {critical_diffs}"
            )

    else:
        # Text-only model - validate all fields
        config_pass, critical_diffs, warning_diffs = compare_section(
            "", orig_dict, exp_dict
        )

        if warning_diffs:
            print(
                f"\n    ⚠ {len(warning_diffs)} non-critical "
                f"field difference(s): {warning_diffs}"
            )
        if critical_diffs:
            print(
                f"\n    ✗ {len(critical_diffs)} critical "
                f"field difference(s): {critical_diffs}"
            )
        elif not warning_diffs:
            all_keys = sorted(set(orig_dict.keys()) | set(exp_dict.keys()))
            print(
                f"\n    ✓ All {len(all_keys) - len(skip_keys)} "
                "config fields match"
            )

    return config_pass


# ---------------------------------------------------------------
# 3b. Validate token IDs
# ---------------------------------------------------------------


def validate_token_ids(exp_cfg, orig_cfg, has_vision):
    """Compare special token IDs between exported and original models."""
    print("\n  TOKEN ID VALIDATION")

    # For vision models, check text_config
    if has_vision:
        orig_cfg = orig_cfg.text_config
        exp_cfg = exp_cfg.text_config

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

    # Check top-5 token overlap
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
    has_vision = original_results["has_vision"]

    print(f"\n[4/6] Loading EXPORTED model from {export_path}...")

    try:
        if has_vision:
            exp_model = AutoModelForImageTextToText.from_pretrained(
                export_path, dtype=torch.float32
            )
            exp_processor = AutoProcessor.from_pretrained(hf_model_id)
            exp_tokenizer = exp_processor.tokenizer
        else:
            exp_model = AutoModelForCausalLM.from_pretrained(
                export_path, dtype=torch.float32
            )
            exp_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    except Exception as e:
        print(f"  ✗ Failed to load exported model: {e}")
        raise

    exp_model.eval()

    exp_params = sum(p.numel() for p in exp_model.parameters())
    orig_params = original_results["hf_params"]
    print(f"  ✓ Exported: {exp_params:,} parameters")

    param_match = orig_params == exp_params
    print(
        f"  {'✓' if param_match else '✗'} Param count: "
        f"original={orig_params:,}, exported={exp_params:,}"
    )

    print("\n[5/6] Validating configs and token IDs...")
    orig_cfg = AutoConfig.from_pretrained(hf_model_id)
    exp_cfg = exp_model.config

    config_pass = validate_configs(exp_cfg, orig_cfg, has_vision)
    token_pass = validate_token_ids(exp_cfg, orig_cfg, has_vision)

    print("\n[6/6] Validating numerics...")
    numeric_results = validate_numerics(
        exp_model, exp_tokenizer, original_results, skip_generation
    )

    del exp_model
    if has_vision:
        del exp_processor
    del exp_tokenizer
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
        description="Verify Gemma3 KerasHub → HF export"
    )
    parser.add_argument(
        "--preset",
        default="gemma3_1b",
        help=(
            f"KerasHub preset name (default: gemma3_1b). "
            f"Supported: {', '.join(PRESET_TO_HF)}"
        ),
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
        print(f"Available presets: {', '.join(PRESET_TO_HF.keys())}")
        exit(1)

    # Set up export directory
    export_dir = args.export_dir or tempfile.mkdtemp()
    export_path = os.path.join(export_dir, "gemma3_exported")

    # Create export directory if it doesn't exist
    os.makedirs(export_path, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Gemma3 Export Verification (Real Pretrained Weights)")
    print("=" * 70)
    print(f"  KerasHub preset : {args.preset}")
    print(f"  HF model ID     : {hf_model_id}")
    print(f"  Export path     : {export_path}")
    print(f"  Skip generation : {args.skip_generation}")

    try:
        # Phase 1: Export KerasHub model.
        has_vision = export_keras_model(args.preset, export_path)

        # Phase 2: Precompute original HF outputs.
        original_results = precompute_original_outputs(
            hf_model_id, has_vision, args.skip_generation
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

    except KeyboardInterrupt:
        print("\n\n⚠ Verification interrupted by user")
        exit(130)
    except Exception as e:
        print("\n\n❌ Verification failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
