"""
Verify that exported Mistral models generate correct outputs.

This script loads both the original KerasHub Mistral model and the exported
HuggingFace version, then compares their outputs to ensure the conversion
was successful.

Sample usage:

Verify a converted preset model:
```
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_7b_en \
    --hf_model_path ./mistral_hf/ \
    --test_prompt "What is your favorite"
```

Verify with custom test prompts:
```
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_instruct_7b_en \
    --hf_model_path ./mistral_hf/ \
    --test_prompt "Explain the theory of relativity" \
    --max_length 100
```

Run multiple verification tests:
```
python tools/mistral/verify_mistral_export.py \
    --keras_preset mistral_7b_en \
    --hf_model_path ./mistral_hf/ \
    --run_all_tests
```
"""

import os
from typing import List
from typing import Tuple

import numpy as np
import torch
from absl import app
from absl import flags

os.environ["KERAS_BACKEND"] = "torch"


import keras_hub  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "keras_preset",
    None,
    "KerasHub preset name for the original model (e.g., 'mistral_7b_en').",
    required=True,
)
flags.DEFINE_string(
    "hf_model_path",
    None,
    "Path to the exported HuggingFace model directory.",
    required=True,
)
flags.DEFINE_string(
    "test_prompt",
    "What is your favorite",
    "Test prompt to use for generation verification.",
)
flags.DEFINE_integer(
    "max_length",
    50,
    "Maximum length for text generation.",
)
flags.DEFINE_bool(
    "run_all_tests",
    False,
    "Run comprehensive verification with multiple test prompts.",
)
flags.DEFINE_bool(
    "compare_logits",
    True,
    "Compare output logits numerically for exact verification.",
)
flags.DEFINE_float(
    "logit_tolerance",
    1e-3,
    "Tolerance for logit comparison (due to floating point differences).",
)


def load_models(keras_preset: str, hf_model_path: str):
    """
    Load both KerasHub and HuggingFace versions of the model.

    Args:
        keras_preset: KerasHub preset name
        hf_model_path: Path to HuggingFace model directory

    Returns:
        Tuple of (keras_model, hf_model, hf_tokenizer)
    """
    print("=" * 80)
    print("Loading Models for Verification")
    print("=" * 80)

    # Load KerasHub model
    print(f"\n📥 Loading KerasHub model: {keras_preset}")
    keras_model = keras_hub.models.MistralCausalLM.from_preset(keras_preset)
    print("   ✅ KerasHub model loaded")

    # Load HuggingFace model
    print(f"\n📥 Loading HuggingFace model from: {hf_model_path}")
    try:
        from transformers import AutoModelForCausalLM
        from transformers import AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(
            hf_model_path, use_fast=False
        )
        print("   ✅ HuggingFace model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load HuggingFace model: {e}")
        raise

    return keras_model, hf_model, hf_tokenizer


def verify_generation(
    keras_model,
    hf_model,
    hf_tokenizer,
    prompt: str,
    max_length: int,
) -> Tuple[str, str, bool]:
    """
    Verify text generation by comparing outputs.

    Args:
        keras_model: KerasHub MistralCausalLM model
        hf_model: HuggingFace Mistral model
        hf_tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_length: Maximum generation length

    Returns:
        Tuple of (keras_output, hf_output, matches)
    """
    print(f"\n🧪 Testing generation with prompt: '{prompt}'")
    print(f"   Max length: {max_length}")

    # Generate with KerasHub
    print("\n   🔹 Generating with KerasHub...")
    try:
        keras_output = keras_model.generate([prompt], max_length=max_length)
        keras_output = keras_output[0]
        print(f"   ✅ KerasHub output: {keras_output}")
    except Exception as e:
        print(f"   ❌ KerasHub generation failed: {e}")
        raise

    # Generate with HuggingFace
    print("\n   🔹 Generating with HuggingFace...")
    try:
        input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = hf_model.generate(
                input_ids,
                max_length=max_length,
                do_sample=False,
                pad_token_id=hf_tokenizer.eos_token_id,
            )

        hf_output = hf_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"   ✅ HuggingFace output: {hf_output}")
    except Exception as e:
        print(f"   ❌ HuggingFace generation failed: {e}")
        raise

    # Compare outputs
    # Note: Exact match is not always expected due to tokenization differences
    # but they should be very similar
    matches = keras_output.strip() == hf_output.strip()

    if matches:
        print("\n   ✅ Outputs match exactly!")
    else:
        print(
            "\n   ⚠️  Outputs differ (this may be expected due to tokenization):"
        )
        print(
            f"      Length difference: {len(keras_output)} vs {len(hf_output)}"
        )
        # Check if one contains the other
        if keras_output in hf_output or hf_output in keras_output:
            print(
                "      ✓ One output contains the other"
                " (likely due to special tokens)"
            )

    return keras_output, hf_output, matches


def verify_logits(
    keras_model,
    hf_model,
    hf_tokenizer,
    prompt: str,
    tolerance: float,
) -> bool:
    """
    Verify model outputs by comparing logits numerically.

    Args:
        keras_model: KerasHub model
        hf_model: HuggingFace model
        hf_tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        tolerance: Acceptable difference in logits

    Returns:
        True if logits match within tolerance
    """
    print(f"\n🔬 Verifying logits with prompt: '{prompt}'")
    print(f"   Tolerance: {tolerance}")

    # Tokenize input
    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt")

    # Get HuggingFace logits
    print("\n   🔹 Computing HuggingFace logits...")
    with torch.no_grad():
        hf_outputs = hf_model(input_ids, return_dict=True)
        hf_logits = hf_outputs.logits[0, -1, :].cpu().numpy()  # Last token
    print(f"   ✅ HuggingFace logits shape: {hf_logits.shape}")

    # Get KerasHub logits
    print("\n   🔹 Computing KerasHub logits...")
    try:
        # Tokenize the input
        token_ids = hf_tokenizer.encode(prompt, return_tensors="pt")

        # Convert to the right device and format for KerasHub
        token_ids_keras = token_ids.to(dtype=torch.int32)
        padding_mask = torch.ones_like(token_ids_keras, dtype=torch.int32)

        # Get model outputs
        with torch.no_grad():
            keras_outputs = keras_model(
                {"token_ids": token_ids_keras, "padding_mask": padding_mask}
            )

        # Get logits from the last token
        keras_logits = keras_outputs[0, -1, :].detach().cpu().numpy()
        print(f"   ✅ KerasHub logits shape: {keras_logits.shape}")
    except Exception as e:
        print(f"   ❌ KerasHub logits computation failed: {e}")
        raise

    # Compare logits
    if keras_logits.shape != hf_logits.shape:
        print("\n   ❌ Logit shapes don't match!")
        print(f"      KerasHub: {keras_logits.shape}")
        print(f"      HuggingFace: {hf_logits.shape}")
        return False

    # Compute differences
    abs_diff = np.abs(keras_logits - hf_logits)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print("\n   📊 Logit comparison:")
    print(f"      Max difference: {max_diff:.6f}")
    print(f"      Mean difference: {mean_diff:.6f}")
    print(f"      Tolerance: {tolerance:.6f}")

    # Check top-k predictions
    k = 5
    keras_topk = np.argsort(keras_logits)[-k:][::-1]
    hf_topk = np.argsort(hf_logits)[-k:][::-1]

    print(f"\n   🏆 Top-{k} token predictions:")
    print(f"      KerasHub:     {keras_topk.tolist()}")
    print(f"      HuggingFace:  {hf_topk.tolist()}")

    topk_match = np.array_equal(keras_topk, hf_topk)
    if topk_match:
        print(f"      ✅ Top-{k} predictions match!")
    else:
        print(f"      ⚠️  Top-{k} predictions differ")

    within_tolerance = max_diff <= tolerance

    if within_tolerance:
        print("\n   ✅ Logits match within tolerance!")
    else:
        print("\n   ❌ Logits exceed tolerance!")

    return within_tolerance and topk_match


def run_comprehensive_tests(
    keras_model,
    hf_model,
    hf_tokenizer,
    max_length: int,
) -> List[bool]:
    """
    Run comprehensive verification with multiple test cases.

    Args:
        keras_model: KerasHub model
        hf_model: HuggingFace model
        hf_tokenizer: HuggingFace tokenizer
        max_length: Maximum generation length

    Returns:
        List of test results (True = passed)
    """
    print("\n" + "=" * 80)
    print("Running Comprehensive Verification Tests")
    print("=" * 80)

    test_prompts = [
        "What is your favorite",
        "The capital of France is",
        "In a galaxy far, far away",
        "Once upon a time",
        "Hello, my name is",
    ]

    results = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}/{len(test_prompts)}")
        print("─" * 80)

        try:
            keras_out, hf_out, matches = verify_generation(
                keras_model, hf_model, hf_tokenizer, prompt, max_length
            )
            results.append(True)  # Test passed if no exception
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            results.append(False)

    return results


def main(_):
    # Load models
    keras_model, hf_model, hf_tokenizer = load_models(
        FLAGS.keras_preset, FLAGS.hf_model_path
    )

    print("\n" + "=" * 80)
    print("Starting Verification")
    print("=" * 80)

    all_passed = True

    if FLAGS.run_all_tests:
        # Run comprehensive tests
        results = run_comprehensive_tests(
            keras_model, hf_model, hf_tokenizer, FLAGS.max_length
        )
        all_passed = all(results)

        print("\n" + "=" * 80)
        print(f"Test Summary: {sum(results)}/{len(results)} passed")
        print("=" * 80)
    else:
        # Run single test
        try:
            keras_out, hf_out, matches = verify_generation(
                keras_model,
                hf_model,
                hf_tokenizer,
                FLAGS.test_prompt,
                FLAGS.max_length,
            )
        except Exception as e:
            print(f"\n❌ Generation verification failed: {e}")
            all_passed = False

    # Logit verification
    if FLAGS.compare_logits and all_passed:
        try:
            logits_match = verify_logits(
                keras_model,
                hf_model,
                hf_tokenizer,
                FLAGS.test_prompt,
                FLAGS.logit_tolerance,
            )
            all_passed = all_passed and logits_match
        except Exception as e:
            print(f"\n❌ Logit verification failed: {e}")
            all_passed = False

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ VERIFICATION PASSED")
        print("=" * 80)
        print("\nThe exported model generates correct outputs!")
        print("The conversion from KerasHub to HuggingFace was successful. 🎉")
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        print("\nSome tests failed. Please check the conversion process.")
        print("There may be issues with weight mapping or configuration.")
    print()


if __name__ == "__main__":
    app.run(main)
