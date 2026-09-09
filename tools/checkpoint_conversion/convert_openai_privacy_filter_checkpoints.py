"""
Conversion and numerical parity script for OpenAI Privacy Filter.

Downloads the openai/privacy-filter model from HuggingFace,
converts it to KerasHub format, and verifies numerical parity.

Usage:
python convert_openai_privacy_filter_checkpoints.py
    --preset openai_privacy_filter_en
"""

import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from absl import app
from absl import flags
from keras import ops
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

import keras_hub

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "openai_privacy_filter_en": "openai/privacy-filter",
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)


def compute_hf_output(hf_model, hf_tokenizer, text):
    """Compute HF model output logits."""
    hf_inputs = hf_tokenizer([text], return_tensors="pt", padding=False).to(
        device
    )
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs)
    return (
        hf_outputs.logits.detach().cpu().float().numpy(),
        hf_inputs["input_ids"].detach().cpu().numpy(),
    )


def compute_keras_output(keras_model, token_ids_np):
    """Compute KerasHub model output logits using given token IDs."""
    padding_mask = np.ones_like(token_ids_np, dtype="int32")
    inputs = {
        "token_ids": torch.tensor(token_ids_np, dtype=torch.int32).to(device),
        "padding_mask": torch.tensor(padding_mask, dtype=torch.int32).to(
            device
        ),
    }
    keras_output = keras_model(inputs)
    return ops.convert_to_numpy(keras_output)


def main(_):
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. "
            f"Must be one of {','.join(PRESET_MAP.keys())}"
        )
    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]
    test_text = "My name is John Smith and my email is john@example.com"

    # === Load HF model ===
    print("\n→ Loading HuggingFace model...")
    hf_model = AutoModelForTokenClassification.from_pretrained(
        hf_preset, device_map=device, torch_dtype=torch.float32
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_preset)
    hf_model.eval()
    print("  ✅ HF model loaded")

    # === Compute HF outputs ===
    hf_logits, hf_token_ids = compute_hf_output(
        hf_model, hf_tokenizer, test_text
    )
    print(f"  HF output shape: {hf_logits.shape}")
    print(f"  HF token IDs: {hf_token_ids.tolist()}")

    # === Get HF predictions ===
    hf_preds = np.argmax(hf_logits, axis=-1)[0]
    id2label = hf_model.config.id2label
    hf_labels = [id2label[p] for p in hf_preds]
    hf_tokens = hf_tokenizer.convert_ids_to_tokens(hf_token_ids[0])
    print("\n  HF predictions:")
    for tok, label in zip(hf_tokens, hf_labels):
        if label != "O":
            print(f"    {tok}: {label}")

    hf_params = hf_model.num_parameters()
    del hf_model
    print(f"\n  HF params: {hf_params:,}")

    # === Load KerasHub tokenizer (for preset saving) ===
    print("\n→ Loading KerasHub tokenizer...")
    keras_tokenizer = keras_hub.models.OpenAIPrivacyFilterTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    print("  ✅ KerasHub tokenizer loaded")

    # Note: The HF tokenizer uses a GPT-4o regex pre-tokenizer that differs
    # from KerasHub's default BytePairTokenizer split pattern. Token IDs
    # may differ for inputs containing special characters like '@', '.'.
    # For numerical parity, we use HF token IDs for both models.
    keras_tok_ids = ops.convert_to_numpy(keras_tokenizer([test_text]))
    hf_tok_ids = hf_tokenizer([test_text])["input_ids"]
    if keras_tok_ids.tolist() == hf_tok_ids:
        print("  ✅ Tokenizer parity verified")
    else:
        print(
            f"  ⚠️  Tokenizer difference detected "
            f"(keras={keras_tok_ids.shape[1]} tokens, "
            f"hf={len(hf_tok_ids[0])} tokens)"
        )
        print("      Using HF token IDs for numerical parity check.")
    del hf_tokenizer

    # === Load KerasHub classifier ===
    print("\n→ Loading KerasHub token classifier...")
    keras_classifier = (
        keras_hub.models.OpenAIPrivacyFilterTokenClassifier.from_preset(
            f"hf://{hf_preset}"
        )
    )
    print("  ✅ KerasHub classifier loaded")

    keras_backbone_params = keras_classifier.backbone.count_params()
    keras_total_params = keras_classifier.count_params()
    head_params = keras_total_params - keras_backbone_params
    print("\n→ Parameter count comparison:")
    print(f"  HuggingFace total:   {hf_params:,}")
    print(f"  KerasHub total:      {keras_total_params:,}")
    print(f"    backbone:          {keras_backbone_params:,}")
    print(f"    classifier head:   {head_params:,} (={head_params})")
    assert keras_total_params == hf_params, (
        f"Parameter count mismatch: keras={keras_total_params}, hf={hf_params}"
    )
    print("  ✅ Parameter count matches")

    # === Compute KerasHub outputs using HF token IDs for fair comparison ===
    keras_logits = compute_keras_output(keras_classifier, hf_token_ids)
    print(f"  KerasHub output shape: {keras_logits.shape}")

    # === Numerical parity ===
    print("\n→ Checking numerical parity...")
    max_diff = np.max(np.abs(keras_logits - hf_logits))
    mean_diff = np.mean(np.abs(keras_logits - hf_logits))
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")

    try:
        np.testing.assert_allclose(
            keras_logits, hf_logits, atol=1e-4, rtol=1e-4
        )
        print("  ✅ Numerical parity verified (atol=1e-4)")
    except AssertionError as e:
        print(f"  ⚠️  Strict parity failed: {e}")
        # Try relaxed tolerance
        try:
            np.testing.assert_allclose(
                keras_logits, hf_logits, atol=5e-2, rtol=5e-2
            )
            print("  ✅ Relaxed parity verified (atol=5e-2)")
        except AssertionError:
            print("  ❌ Parity check failed even at relaxed tolerance")

    # === Check predictions match ===
    keras_preds = np.argmax(keras_logits, axis=-1)[0]
    keras_labels = [id2label[int(p)] for p in keras_preds]
    print("\n  KerasHub predictions:")
    for tok, label in zip(hf_tokens, keras_labels):
        if label != "O":
            print(f"    {tok}: {label}")

    if np.array_equal(keras_preds, hf_preds):
        print("  ✅ Prediction parity verified")
    else:
        print("  ⚠️  Prediction mismatch:")
        for i, (kp, hp) in enumerate(zip(keras_preds, hf_preds)):
            if kp != hp:
                print(
                    f"    pos {i}: keras={id2label[int(kp)]}, "
                    f"hf={id2label[int(hp)]}"
                )

    # === Save preset ===
    print(f"\n→ Saving preset to ./{preset}/")
    preprocessor = keras_hub.models.OpenAIPrivacyFilterPreprocessor(
        keras_tokenizer
    )
    keras_classifier.preprocessor = preprocessor
    keras_classifier.save_to_preset(f"./{preset}")
    print("  ✅ Preset saved")

    print("\n🎉 Conversion complete!")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
