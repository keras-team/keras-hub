"""
Verify that a KerasHub Llama3 export matches the original HuggingFace model.

This script loads both a KerasHub Llama3 preset and a HuggingFace model
directory (produced by export_llama3_to_hf.py or another conversion tool)
and compares their outputs to confirm the conversion is numerically correct.

Sample usage:

Basic logit comparison with a single prompt:
```
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_8b_en \
    --hf_model_path ./llama3_hf/
```

Full verification with text generation comparison:
```
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_instruct_8b_en \
    --hf_model_path ./llama3_instruct_hf/ \
    --test_prompt "What is the capital of France?" \
    --max_length 100 \
    --run_all_tests
```

Adjust numerical tolerance:
```
python tools/llama3/verify_llama3_export.py \
    --keras_preset llama3_8b_en \
    --hf_model_path ./llama3_hf/ \
    --logit_tolerance 1e-2
```
"""

import os
import sys

from absl import app
from absl import flags

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np  # noqa: E402
import torch  # noqa: E402

import keras_hub  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "keras_preset",
    None,
    "KerasHub Llama3 preset name (e.g., 'llama3_8b_en'). Required.",
)
flags.DEFINE_string(
    "hf_model_path",
    None,
    "Path to the HuggingFace model directory to compare against. "
    "Should be the output directory from export_llama3_to_hf.py. Required.",
)
flags.DEFINE_string(
    "test_prompt",
    "The quick brown fox jumps over the lazy dog",
    "Prompt to use for generation and logit comparison tests.",
)
flags.DEFINE_integer(
    "max_length",
    50,
    "Maximum generation length (in tokens) for text generation tests.",
)
flags.DEFINE_bool(
    "run_all_tests",
    False,
    "If True, run a suite of test prompts instead of a single prompt.",
)
flags.DEFINE_bool(
    "compare_logits",
    True,
    "If True, compare logits numerically between KerasHub and HF models.",
)
flags.DEFINE_float(
    "logit_tolerance",
    1e-3,
    "Maximum absolute difference allowed between corresponding logits.",
)

# Mark required flags
flags.mark_flag_as_required("keras_preset")
flags.mark_flag_as_required("hf_model_path")


def load_models(keras_preset: str, hf_model_path: str):
    """
    Load both KerasHub and HuggingFace Llama3 models.

    Returns:
        Tuple of (keras_model, hf_model, hf_tokenizer)
    """
    from transformers import AutoTokenizer
    from transformers import LlamaForCausalLM

    print(f"\n📥 Loading KerasHub Llama3 model: {keras_preset}")
    keras_model = keras_hub.models.Llama3CausalLM.from_preset(keras_preset)
    print("   ✅ KerasHub model loaded")

    print(f"\n📥 Loading HuggingFace model from: {hf_model_path}")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    hf_model = LlamaForCausalLM.from_pretrained(
        hf_model_path, torch_dtype="auto"
    )
    hf_model.eval()
    print("   ✅ HuggingFace model loaded")

    return keras_model, hf_model, hf_tokenizer


def verify_generation(
    keras_model,
    hf_model,
    hf_tokenizer,
    prompt: str,
    max_length: int,
):
    """
    Compare text generation between KerasHub and HuggingFace models.

    Returns:
        Tuple of (keras_output, hf_output) strings.
    """
    print(f"\n{'─' * 60}")
    print(f"📝 Test prompt: {repr(prompt)}")
    print(f"{'─' * 60}")

    # KerasHub generation
    print("\n🔵 KerasHub generation:")
    keras_output = keras_model.generate([prompt], max_length=max_length)
    if isinstance(keras_output, list):
        keras_text = keras_output[0]
    else:
        keras_text = str(keras_output)
    print(f"   {repr(keras_text)}")

    # HuggingFace generation
    print("\n🟠 HuggingFace generation:")
    hf_inputs = hf_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        hf_output_ids = hf_model.generate(
            **hf_inputs,
            max_length=max_length,
            do_sample=False,
            temperature=1.0,
            pad_token_id=hf_tokenizer.eos_token_id,
        )
    hf_text = hf_tokenizer.decode(hf_output_ids[0], skip_special_tokens=True)
    print(f"   {repr(hf_text)}")

    # Check if outputs match
    match = keras_text.strip() == hf_text.strip()
    if match:
        print("\n✅ Generation outputs MATCH")
    else:
        print("\n⚠️  Generation outputs DIFFER")
        # Compute token-level prefix match
        keras_tokens = hf_tokenizer.encode(keras_text)
        hf_tokens = hf_tokenizer.encode(hf_text)
        prefix_match = 0
        for a, b in zip(keras_tokens, hf_tokens):
            if a == b:
                prefix_match += 1
            else:
                break
        total = min(len(keras_tokens), len(hf_tokens))
        print(f"   Token prefix match: {prefix_match}/{total}")

    return keras_text, hf_text


def verify_logits(
    keras_model,
    hf_model,
    hf_tokenizer,
    prompt: str,
    tolerance: float,
):
    """
    Compare forward-pass logits between KerasHub and HuggingFace models.

    Encodes the prompt, runs a single forward pass through each model,
    and compares the logits at the last token position.

    Returns:
        True if logits are within tolerance, False otherwise.
    """
    print(f"\n{'─' * 60}")
    print("🔬 Logit comparison")
    print(f"   Prompt: {repr(prompt)}")
    print(f"   Tolerance: {tolerance}")
    print(f"{'─' * 60}")

    # Tokenize
    hf_inputs = hf_tokenizer(prompt, return_tensors="pt")
    input_ids = hf_inputs["input_ids"]  # (1, seq_len)
    seq_len = input_ids.shape[1]
    print(f"   Sequence length: {seq_len} tokens")

    # HuggingFace forward pass
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    hf_logits = hf_outputs.logits[0, -1, :]  # (vocab_size,)
    hf_logits_np = hf_logits.detach().cpu().to(torch.float32).numpy()

    # KerasHub forward pass
    token_ids_np = input_ids.numpy()  # (1, seq_len)
    padding_mask_np = np.ones_like(token_ids_np)

    token_ids_tensor = torch.tensor(token_ids_np)
    padding_mask_tensor = torch.tensor(padding_mask_np)

    # Run through the backbone (logits output)
    with torch.no_grad():
        keras_outputs = keras_model.backbone(
            {
                "token_ids": token_ids_tensor,
                "padding_mask": padding_mask_tensor,
            }
        )
    # Project to vocab: backbone output × reverse_embedding
    embeddings = keras_model.backbone.token_embedding
    keras_logits = embeddings(
        keras_outputs, reverse=True
    )  # (1, seq_len, vocab)
    keras_logits_last = keras_logits[0, -1, :]  # (vocab_size,)
    keras_logits_np = keras_logits_last.detach().cpu().to(torch.float32).numpy()

    # Compute difference
    abs_diff = np.abs(keras_logits_np - hf_logits_np)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    print(f"\n   Max absolute logit difference:  {max_diff:.6f}")
    print(f"   Mean absolute logit difference: {mean_diff:.6f}")
    print(f"   Tolerance:                      {tolerance:.6f}")

    # Compare top-5 predictions
    keras_top5 = np.argsort(keras_logits_np)[::-1][:5]
    hf_top5 = np.argsort(hf_logits_np)[::-1][:5]
    print("\n   Top-5 predicted tokens:")
    print(f"   {'Rank':<6} {'KerasHub':<15} {'HuggingFace':<15} {'Match'}")
    print(f"   {'─' * 5:<6} {'─' * 14:<15} {'─' * 14:<15} {'─' * 5}")
    for rank, (k_idx, h_idx) in enumerate(zip(keras_top5, hf_top5), 1):
        k_tok = hf_tokenizer.decode([k_idx])
        h_tok = hf_tokenizer.decode([h_idx])
        match_sym = "✅" if k_idx == h_idx else "❌"
        print(f"   {rank:<6} {repr(k_tok):<15} {repr(h_tok):<15} {match_sym}")

    passed = max_diff <= tolerance
    if passed:
        print(
            f"\n✅ Logit comparison PASSED "
            f"(max diff {max_diff:.2e} ≤ {tolerance:.2e})"
        )
    else:
        print(
            f"\n❌ Logit comparison FAILED "
            f"(max diff {max_diff:.2e} > {tolerance:.2e})"
        )

    return passed


def run_comprehensive_tests(
    keras_model,
    hf_model,
    hf_tokenizer,
    max_length: int,
    tolerance: float,
    compare_logits: bool,
):
    """Run a suite of test prompts to verify the export."""
    test_prompts = [
        "The quick brown fox jumps over the lazy dog",
        "What is the capital of France?",
        "In machine learning, a neural network is",
        "Once upon a time in a land far away",
        "The theory of relativity states that",
    ]

    print("\n" + "=" * 80)
    print("🧪 Running comprehensive test suite")
    print(f"   {len(test_prompts)} prompts, max_length={max_length}")
    print("=" * 80)

    generation_results = []
    logit_results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}/{len(test_prompts)}]")

        keras_text, hf_text = verify_generation(
            keras_model, hf_model, hf_tokenizer, prompt, max_length
        )
        generation_results.append(keras_text.strip() == hf_text.strip())

        if compare_logits:
            passed = verify_logits(
                keras_model, hf_model, hf_tokenizer, prompt, tolerance
            )
            logit_results.append(passed)

    # Summary
    print("\n" + "=" * 80)
    print("📊 Comprehensive Test Summary")
    print("=" * 80)

    gen_pass = sum(generation_results)
    gen_total = len(generation_results)
    print(
        f"\n  Generation match: {gen_pass}/{gen_total} prompts "
        f"({'✅ PASS' if gen_pass == gen_total else '⚠️ PARTIAL'})"
    )

    if logit_results:
        logit_pass = sum(logit_results)
        logit_total = len(logit_results)
        print(
            f"  Logit comparison: {logit_pass}/{logit_total} prompts "
            f"({'✅ PASS' if logit_pass == logit_total else '❌ FAIL'})"
        )

    overall_pass = gen_pass == gen_total and (
        not logit_results or sum(logit_results) == len(logit_results)
    )
    print(
        f"\n  Overall result: "
        f"{'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}"
    )


def main(_):
    if not os.path.isdir(FLAGS.hf_model_path):
        print(
            "Error: HuggingFace model path does not exist: "
            f"{FLAGS.hf_model_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Llama3 Export Verification: KerasHub vs HuggingFace")
    print("=" * 80)
    print(f"\n  KerasHub preset : {FLAGS.keras_preset}")
    print(f"  HF model path   : {FLAGS.hf_model_path}")
    print(f"  Logit tolerance : {FLAGS.logit_tolerance}")
    print(f"  Max length      : {FLAGS.max_length}")
    print(f"  Compare logits  : {FLAGS.compare_logits}")
    print(f"  Run all tests   : {FLAGS.run_all_tests}")

    keras_model, hf_model, hf_tokenizer = load_models(
        FLAGS.keras_preset, FLAGS.hf_model_path
    )

    if FLAGS.run_all_tests:
        run_comprehensive_tests(
            keras_model,
            hf_model,
            hf_tokenizer,
            max_length=FLAGS.max_length,
            tolerance=FLAGS.logit_tolerance,
            compare_logits=FLAGS.compare_logits,
        )
    else:
        verify_generation(
            keras_model,
            hf_model,
            hf_tokenizer,
            FLAGS.test_prompt,
            FLAGS.max_length,
        )

        if FLAGS.compare_logits:
            passed = verify_logits(
                keras_model,
                hf_model,
                hf_tokenizer,
                FLAGS.test_prompt,
                FLAGS.logit_tolerance,
            )
            if not passed:
                sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ Verification complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    app.run(main)
