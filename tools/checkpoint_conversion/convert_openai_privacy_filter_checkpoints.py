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
flags.DEFINE_string(
    "dtype",
    "bfloat16",
    "The dtype for the Keras model and the saved preset.",
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
        hf_preset, device_map=device, dtype=torch.float32
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
    print(f"\n  HF params: {hf_params:,}")

    # === Load KerasHub tokenizer and preprocessor ===
    print("\n→ Loading KerasHub tokenizer and preprocessor...")
    keras_tokenizer = keras_hub.models.OpenAIPrivacyFilterTokenizer.from_preset(
        f"hf://{hf_preset}"
    )
    preprocessor = keras_hub.models.OpenAIPrivacyFilterPreprocessor(
        keras_tokenizer
    )
    print("  ✅ KerasHub tokenizer and preprocessor loaded")

    corpus = [
        "My name is John Smith and my email is john@example.com",
        "card 4111 1111 1111 1111",
        "Call +1 (555) 123-4567 or 555.987.6543",
        "SSN 123-45-6789, account 00012345678, routing 021000021",
        "Contact me at jane.doe+work@sub.example.co.uk today.",
        "See https://example.com/path?user=bob&id=12345 for details",
        "Born 12/25/1990, moved on March 3rd, 2015.",
        "Zoë Müller lives at Straße 12, 80331 München",
        "IP 192.168.1.254 and MAC 00:1A:2B:3C:4D:5E",
        "IBAN DE89 3704 0044 0532 0130 00",
        "tabs\there\tand  double  spaces   triple",
        "123 Main St., Apt 4B, Springfield, IL 62704",
    ]

    mismatches = []
    for text in corpus:
        hf_ids = hf_tokenizer(text)["input_ids"]
        out = preprocessor([text])
        keras_ids = np.asarray(out["token_ids"])[0]
        padding_mask = np.asarray(out["padding_mask"])[0]
        keras_ids = keras_ids[padding_mask.astype(bool)].tolist()
        if hf_ids != keras_ids:
            mismatches.append(
                f"Text: {text!r}\n  HF: {hf_ids}\n  Keras: {keras_ids}"
            )

    if mismatches:
        raise AssertionError(
            "Preprocessor parity check failed on "
            f"{len(mismatches)}/{len(corpus)} strings:\n"
            + "\n".join(mismatches)
        )
    print("  ✅ Preprocessor parity verified")

    # === Load KerasHub classifier ===
    print("\n→ Loading KerasHub token classifier...")
    keras_classifier = (
        keras_hub.models.OpenAIPrivacyFilterTokenClassifier.from_preset(
            f"hf://{hf_preset}", dtype=FLAGS.dtype
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
    if keras_total_params != hf_params:
        raise ValueError(
            "Parameter count mismatch: "
            f"keras={keras_total_params}, hf={hf_params}"
        )
    print("  ✅ Parameter count matches")

    # === Save preset ===
    print(f"\n→ Saving preset to ./{preset}/")
    keras_classifier.preprocessor = preprocessor
    keras_classifier.save_to_preset(f"./{preset}")
    print("  ✅ Preset saved")

    # === Check predictions match on varying lengths ===
    print("\n→ Checking predict([text]) parity...")
    # Load back the saved preset to ensure it works
    loaded_classifier = (
        keras_hub.models.OpenAIPrivacyFilterTokenClassifier.from_preset(
            f"./{preset}"
        )
    )
    # The check is about parity; increase sequence_length so we do not truncate.
    loaded_classifier.preprocessor.sequence_length = 1024

    test_texts = [
        "My name is John Smith and my email is john@example.com",
        "John Smith works at Example Corp. " * 30,  # ~210 tokens
        "Here is another much longer sentence to test. " * 68,  # ~610 tokens
    ]

    for i, text in enumerate(test_texts):
        hf_logits, hf_token_ids = compute_hf_output(
            hf_model, hf_tokenizer, text
        )
        hf_preds = np.argmax(hf_logits, axis=-1)[0]
        valid_len = len(hf_preds)
        print(f"\n  Text {i + 1} length: {valid_len} tokens")

        keras_logits = loaded_classifier.predict([text], verbose=0)
        # We only care about unpadded tokens.
        keras_valid_logits = keras_logits[0][:valid_len]
        keras_preds = np.argmax(keras_valid_logits, axis=-1)
        hf_valid_logits = hf_logits[0][:valid_len]

        max_diff = np.max(np.abs(keras_valid_logits - hf_valid_logits))
        mean_diff = np.mean(np.abs(keras_valid_logits - hf_valid_logits))
        print(f"    Max absolute diff: {max_diff:.6f}")
        print(f"    Mean absolute diff: {mean_diff:.6f}")

        if FLAGS.dtype == "float32":
            np.testing.assert_allclose(
                keras_valid_logits, hf_valid_logits, atol=1e-4, rtol=1e-4
            )

        if not np.array_equal(keras_preds, hf_preds):
            raise ValueError(f"Prediction mismatch for text length {valid_len}")

    print("  ✅ Prediction parity verified for all lengths")
    print("\n🎉 Conversion complete!")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
